import os
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SIGMOID_PARAMS_IMAGENET_SC = {"ViT-SO400M-14-SigLIP-384":{"a":58, "b":-0.13},
                              "ViT-L-16-SigLIP-384":{"a":60, "b":-0.09},
                              "ViT-L-14-336":{"a":58, "b":-0.25},
                              }

#for standard models, using OpenAI weights if available, otherwise OpenCLIP model
CN_TO_CHECKPOINT = {"ViT-SO400M-14-SigLIP-384": "webli",
                    "ViT-L-16-SigLIP-384": "webli",
                    "ViT-g-14": "laion2b_s34b_b88k",
                    "ViT-L-14-336": "openai",
                    "ViT-B-16": "openai",
                    "ViT-L-16-SigLIP-256": "webli",
                    "ViT-B-16-SigLIP-384": "webli",
                    }

EXPLANATION_PATHS = {"le_label":"results/ours/LE_label/",
                     "le_siglip":"results/ours/LE_siglip/",
                     "net_dissect":"results/baselines/NetDissect/",
                     "milan":"results/baselines/MILAN/",
                     "clip_dissect":"results/baselines/CLIP_Dissect/"}

def save_activations(model, dataset, device, target_layer, target_neuron, batch_size, save_path):
    """
    saves full activations of a specific neuron
    """
    act_path = os.path.join(save_path, 'act.pt')
    if os.path.exists(act_path):
        activations = torch.load(act_path, map_location = device)
        return activations.float()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    activations = get_target_activations(model, dataset, [target_layer], start=target_neuron, end=target_neuron+1,
                                          batch_size=batch_size, device=device, pool_mode='none')
    activations = torch.cat(activations[target_layer]).squeeze()
    torch.save(activations.half(), act_path)
    return activations.to(device)

def save_summary_activations(model, dataset, device, target_layer, batch_size, save_path, pool_mode="avg"):
    act_path = os.path.join(save_path, 'all_{}.pt'.format(pool_mode))
    if os.path.exists(act_path):
        activations = torch.load(act_path, map_location = device)
        return activations.float()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    activations = get_target_activations(model, dataset, [target_layer], batch_size=batch_size, device=device, pool_mode=pool_mode)
    activations = torch.cat(activations[target_layer])
    torch.save(activations.half(), act_path)
    return activations.to(device)

def get_target_activations(target_model, dataset, target_layers = ["layer4"], start=None, end=None, batch_size = 128,
                            device = "cuda", pool_mode="none"):
   
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation_slice(all_features[target_layer], pool_mode, start, end))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
            
    for target_layer in target_layers:
        hooks[target_layer].remove()
            
    return all_features

def save_clip_image_features(model, dataset, save_name, batch_size=128, device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=256):
    if os.path.exists(save_name):
        return torch.load(save_name)
    _make_save_dir(save_name)
    feats = get_clip_text_features(model, text, batch_size)
    torch.save(feats, save_name)
    return feats

def get_clip_text_features(model, text, batch_size=256):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in range(math.ceil(len(text)/batch_size)):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_activation_slice(outputs, mode, start=None, end=None):
    '''
    start, end: the endpoints of neurons to record
    mode: how to pool activations: one of avg, max, first, none
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output[:, start:end].mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==3:
                outputs.append(output[:, :, start:end].mean(dim=[1]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output[:, start:end].detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output[:, start:end].amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==3:
                outputs.append(output[:, :, start:end].amax(dim=[1]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output[:, start:end].detach().cpu())
    elif mode=='first':
        # only record CLS token for ViT last layer. Note ViT has different order of inputs
        def hook(model, input, output):
            if len(output.shape)==3:
                outputs.append(output[:, 0, start:end].detach().cpu())
    elif mode=='none':
        def hook(model, input, output):
            if len(output.shape)==3:
                outputs.append(output[:, :, start:end].detach().cpu())
            else:
                outputs.append(output[:, start:end].detach().cpu())
    return hook

def save_pred_acc_loss(model, dataset, device, save_path, batch_size=128, T=1.1, substitute_activations=None):
    """
    save_path: directory to save results in
    T: temperature used for calibration
    Also saves_results without calib, i.e. T=1, but only returns loss with selected T
    """
    preds_path = os.path.join(save_path, 'preds.pt')
    accs_path = os.path.join(save_path, 'accs.pt')
    losses_path = os.path.join(save_path, 'losses_T_{:.2f}.pt'.format(T))
    losses_no_calib_path = os.path.join(save_path, 'losses_T_{:.2f}.pt'.format(1))
    
    if os.path.exists(save_path):
        try:
            preds = torch.load(preds_path, map_location = device)
            accs = torch.load(accs_path, map_location = device).float()
            losses = torch.load(losses_path, map_location = device)
            return preds, accs, losses
        except(FileNotFoundError):
            pass
    else:
        os.makedirs(save_path)
        
    with torch.no_grad():
        preds = []
        accs = []
        losses = []
        losses_no_calib = []
        
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        
        for i, (images, labels) in enumerate(DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)):
            with torch.no_grad():
                if substitute_activations is not None:
                    outs_no_calib = model(images.to(device), substitute_activations[i*batch_size:(i+1)*batch_size])
                else:
                    outs_no_calib = model(images.to(device))
                outs = outs_no_calib/T
                pred = torch.argmax(outs, dim=1)
                acc = (pred==labels.to(device))
                loss = loss_fn(outs, labels.to(device))
                loss_no_calib = loss_fn(outs_no_calib, labels.to(device))
                preds.append(pred)
                accs.append(acc)
                losses.append(loss)
                losses_no_calib.append(loss_no_calib)
                
        preds = torch.cat(preds, dim=0)
        accs = torch.cat(accs, dim=0)
        losses = torch.cat(losses, dim=0)
        losses_no_calib = torch.cat(losses_no_calib, dim=0)
        
        torch.save(preds, preds_path)
        torch.save(accs, accs_path)
        torch.save(losses, losses_path)
        if T!=1:
            torch.save(losses_no_calib, losses_no_calib_path)
        return preds, accs.float(), losses

def save_pal_without_neuron(model, dataset, device, target_layer, target_neuron, save_path, batch_size=128, T=1):
    # be careful, modifies model in place
    def new_forward(self, x):
        x = self.conv1(x)
        if target_layer=="conv1":
            x[:, target_neuron] *= 0
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if target_layer=="maxpool":
            x[:, target_neuron] *= 0
        x = self.layer1(x)
        if target_layer=="layer1":
            x[:, target_neuron] *= 0
        x = self.layer2(x)
        if target_layer=="layer2":
            x[:, target_neuron] *= 0
        x = self.layer3(x)
        if target_layer=="layer3":
            x[:, target_neuron] *= 0
        x = self.layer4(x)
        if target_layer=="layer4":
            x[:, target_neuron] *= 0
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if target_layer=="fc":
            raise ValueError
        if target_layer=="softmax":
            x = torch.nn.functional.softmax(x, dim=1)
            raise ValueError
        return x

    model.forward = new_forward.__get__(model)
    return save_pred_acc_loss(model, dataset, device, save_path, batch_size=batch_size, T=T)

def get_per_neuron_impact(orig_acc, orig_loss, new_acc, new_loss):
    """
    Returns a tensor of per input neuron impact, as a fraction
    Sum of this is tensor a percentage of how helpful that neuron is to the network
    sum of 0.1 means removing the neuron drops the network overall performance
    by 10%, measured as the average of drop in accuracy and increase in loss
    """
    acc_effect = (orig_acc-new_acc)/torch.sum(orig_acc)
    loss_effect = -(orig_loss-new_loss)/torch.sum(orig_loss)
    return (acc_effect+loss_effect)/2
