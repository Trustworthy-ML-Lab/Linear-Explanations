import json
import numpy as np
import torch
import open_clip

import utils
import data_utils

from torch.utils.data import DataLoader, TensorDataset
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga

def predict_activations(description_df, target_neuron, clip_image_features, clip_model, tokenizer, a, b, device="cuda"):
    descriptions = []
    curr_linear_weights = []
    curr_df = description_df[description_df["unit"]==target_neuron]
    if "description" in curr_df.columns: # simple explanation
        descriptions.append(curr_df["description"].iloc[0])
    else: #linear explanation
        for i in range(100):
            try:
                concept = curr_df["concept{}".format(i)].iloc[0]
                if str(concept) == "0":
                    break
                else:
                    descriptions.append(concept)
                    curr_linear_weights.append(float(curr_df["weight{}".format(i)].iloc[0]))
            except(KeyError):
                break

    tokenized_descriptions = tokenizer(descriptions).to(device)

    clip_text_features = utils.get_clip_text_features(clip_model, tokenized_descriptions).float()
    with torch.no_grad():
        clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)
        clip_feats = (clip_image_features @ clip_text_features.T)
        clip_feats = torch.nn.functional.sigmoid(a*(clip_feats+b))
    
    if len(curr_linear_weights) == 0: #simple explanation
        preds = clip_feats
    else: #linear explanations
        weights = torch.tensor(curr_linear_weights).to(device)
        weights = weights.squeeze().unsqueeze(0)
        preds = torch.sum(clip_feats*weights, dim=1, keepdims=True)+float(curr_df["bias"].iloc[0])
    return preds

def get_target_acts(target_name, dataset_name, target_layer, save_dir, batch_size, device, start_neuron=0, end_neuron=None, pool_mode="avg"):
    model, preprocess = data_utils.get_target_model(target_name, device=device)
    target_data = data_utils.get_data(dataset_name, preprocess)
    
    layer_save_path = '{}/{}_{}/{}/'.format(save_dir, target_name, dataset_name, target_layer)
    activations = utils.save_summary_activations(model, target_data, device, target_layer, batch_size, layer_save_path, pool_mode)
    summary_activations = activations[:, start_neuron:end_neuron]
    return summary_activations

def get_clip_feats(clip_name, dataset_name, concept_set, save_dir,  batch_size, device):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_name.split("_")[-1],
                                                                       pretrained=utils.CN_TO_CHECKPOINT[clip_name],
                                                                       device=device)
    clip_data = data_utils.get_data(dataset_name, clip_preprocess)
    clip_save_name = "{}/{}_{}.pt".format(save_dir, dataset_name, clip_name)
    utils.save_clip_image_features(clip_model, clip_data, clip_save_name, batch_size, device)
    clip_image_features = torch.load(clip_save_name, map_location=device).float()
    
    with open(concept_set, 'r') as f: 
        concept_text = (f.read()).split('\n')
    tokenized_text = open_clip.get_tokenizer(clip_name.split("_")[-1])(concept_text).to(device)
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set.split("/")[-1].split(".")[0], clip_name)
    clip_text_features = utils.save_clip_text_features(clip_model, tokenized_text, text_save_name).float()
    a = utils.SIGMOID_PARAMS_IMAGENET_SC[clip_name]["a"]
    b = utils.SIGMOID_PARAMS_IMAGENET_SC[clip_name]["b"]
    
    with torch.no_grad():
        clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
        clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)
        clip_feats = clip_image_features @ (clip_text_features).T
        clip_feats = torch.nn.functional.sigmoid(a*(clip_feats+b))
    return clip_feats

def get_onehot_labels(dataset_name, device):
    """
    Returns:
    onehot_labels: concept activation matrix (|D| x n_concepts)
    concept_text: list of strings with concept names
    """
    concept_set = "data/concept_sets/{}_labels_clean.txt".format(dataset_name.split("_")[0])
    with open(concept_set, 'r') as f: 
        concept_text = (f.read()).split('\n')
    
    dataset = data_utils.get_data(dataset_name)
    num_classes = max(dataset.targets) + 1
    # Convert to one-hot encoding
    onehot_labels = torch.zeros(len(dataset.targets), num_classes)
    onehot_labels[torch.arange(len(dataset.targets)), dataset.targets] = 1
    onehot_labels = onehot_labels.to(device)

    if dataset_name in ("imagenet_val", "cifar100_val"):
        with open('data/concept_sets/{}_superclass_to_ids.json'.format(dataset_name[:-4]), 'r') as f:
            superclass_to_id = json.load(f)
        new_labels = []
        for sclass in superclass_to_id.keys():
            concept_text.append(sclass.replace("_", " "))
            subclasses = superclass_to_id[sclass]
            new_labels.append(torch.sum(torch.stack([onehot_labels[:, i] for i in subclasses], dim=0), dim=0))
        new_labels = torch.stack(new_labels, dim=1)
        onehot_labels = torch.cat([onehot_labels, new_labels], dim=1)

    return onehot_labels, concept_text

def get_glm_datasets(concept_activations, target_activations, train_ids, val_ids):
    train_data = concept_activations[train_ids].float()
    val_data = concept_activations[val_ids].float()
    
    train_target = target_activations[train_ids]
    val_target = target_activations[val_ids]

    with torch.no_grad():
        train_mean = torch.mean(train_data, dim=0, keepdim=True)
        train_std = torch.std(train_data, dim=0, keepdim=True)
        
        train_data -= train_mean
        train_data /= train_std
    
        target_mean = torch.mean(target_activations, dim=0, keepdim=True)
        target_std = torch.std(target_activations, dim=0, keepdim=True)
        train_target = (train_target-target_mean)/target_std
        
        indexed_train_ds = IndexedTensorDataset(train_data, train_target)
    
        val_data -= train_mean
        val_data /= train_std
    
        val_target = (val_target - target_mean)/target_std
        
        val_ds = TensorDataset(val_data, val_target)
        
    return indexed_train_ds, val_ds

def train_glm_model(train_data, val_data, device, saga_batch_size=512, n_iters=500, lam=5e-2, step_size=5e-4, alpha=0.99):
    
    indexed_train_loader = DataLoader(train_data, batch_size=saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=saga_batch_size, shuffle=False)
    n_concepts = len(train_data[0][0])
    n_neurons = len(train_data[0][1])
    
    linear = torch.nn.Linear(in_features = n_concepts, out_features=n_neurons).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = lam
    
    output_proj = glm_saga(linear, indexed_train_loader, step_size, n_iters, alpha, epsilon=1, k=1, val_loader = val_loader,
                           n_ex=len(train_data), do_zero=False, metadata=metadata, n_classes = n_neurons, family='gaussian',
                           lookbehind=5, tol=1e-5*n_neurons, verbose=False)
    
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    linear.load_state_dict({"weight":W_g, "bias":b_g})
    return linear

def train_linear_model(train_data, val_data, train_target, val_target, device):
    new_linear = torch.nn.Linear(in_features = train_data.shape[1], out_features = 1).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(new_linear.parameters(), lr=3e-1, betas=(0.9, 0.999), eps=1e-8)
    
    for epoch in range(100):
        simulated = new_linear(train_data)
        loss = loss_fn(simulated, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        simulated_val = new_linear(val_data)
    corr_coefficient = np.corrcoef(val_target.cpu()[:, 0], simulated_val[:, 0].cpu().detach())[0, 1]
    return(corr_coefficient, new_linear)


def greedy_search(train_concept_act, val_concept_act, train_target, val_target, top_concepts, device, 
                  max_length=10, tolerance=0.02, width=10):
    """
    train_concept_act: |D_train| x n_concepts, P(c_j|x_i), not normalized
    val_concept_act: |D_val| x n_concepts
    train_target: |D_train| x 1, target neuron activations, not normalized
    val_target: |D_val| x 1
    top_concepts: |n_concepts|, descending list of highest weight concepts for this neuron
    device: str, which torch device
    max_length: int, maximum length of description
    tolerance: minimum increase in correlation to keep adding a concept
    width: how many concepts to check at each step
    """
    selected_concepts = []
    bad_concepts = []
    best_corr = 0
    best_weight = None
    best_bias = None
    
    while len(selected_concepts) < max_length:
        candidate_concepts = []
        for i in range(len(top_concepts)):
            if i not in selected_concepts and i not in bad_concepts:
                candidate_concepts.append(i)
            if len(candidate_concepts) >= width:
                break
        curr_best_concept = None
        curr_best_corr = best_corr
        for concept in candidate_concepts:
            concept_set = torch.tensor(selected_concepts + [concept])
            concepts = top_concepts[concept_set]
            train_data = train_concept_act[:, concepts]
            val_data = val_concept_act[:, concepts]
            
            corr, new_linear = train_linear_model(train_data, val_data, train_target, val_target, device)
            # if adding this concept does not improve correlation by more than tol now, never use it
            # second condition to make sure all explanations are at least length 1 even if low correlation
            if (corr - best_corr < tolerance) and (len(selected_concepts)>0): 
                bad_concepts.append(concept)
            elif corr > curr_best_corr:
                curr_best_corr = corr
                curr_best_concept = concept
                best_bias = float(new_linear.bias)
                best_weight = new_linear.weight[0].detach()
        if curr_best_concept != None:
            best_corr = curr_best_corr
            selected_concepts.append(curr_best_concept)
            
        else:
            break
    concepts = top_concepts[concept_set]
    return concepts, best_weight, best_bias, best_corr
    