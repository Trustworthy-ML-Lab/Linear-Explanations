import argparse
import os
import math
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
import data_utils
import linear_explanation


def parse_arguments():
    parser = argparse.ArgumentParser(description='Linear explanations')
    parser.add_argument('--mode', type=str, default='label', help='which version to use label or siglip')
    parser.add_argument('--device', type=str, default='cuda', help='whether to use gpu')
    parser.add_argument('--dataset_name', type=str, default='imagenet_val', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for clip and target model')
    parser.add_argument('--activations_dir', type=str, default='saved_activations', help='Save directory')
    parser.add_argument('--result_dir', type=str, default='results', help='Save directory')
    parser.add_argument('--clip_name', type=str, default='ViT-L-16-SigLIP-384', help='Which CLIP model to use')
    parser.add_argument('--concept_set', type=str, default=None, help='Concept set path, only used in siglip mode. If not provided uses default based on dataset.')
    parser.add_argument('--target_name', type=str, default='resnet50_imagenet', help='Target model')
    parser.add_argument('--target_layer', type=str, default='layer4', help='Target layer')
    parser.add_argument('--start_neuron', type=int, default=0, help='First neuron')
    parser.add_argument('--end_neuron', type=int, default=None, help='Last neuron, not included. Default is all neurons.')
    parser.add_argument('--max_length', type=int, default=10, help='Maximum description length')
    parser.add_argument('--tolerance', type=float, default=0.02, help='''Minimum increase in correlation to include a new concept.
                                                                         Smaller tolerance -> longer explanation.''')
    parser.add_argument('--glm_neuron_batch', type=int, default=128, help="How many neurons we optimize for in one pass of GLM-Saga")
    parser.add_argument('--glm_lambda', type=float, default = 0.05, help="Parameter controlling sparsity of the initial weights. Lowering often improves stability.")
    parser.add_argument('--pool_mode', type=str, default='avg', help="Activation pooling function for i.e. CNN channel outputs. {avg, max, first}")
    return parser.parse_args()

def explain_neurons(args):
    if args.mode == "siglip":
        if args.concept_set == None:
            concept_set = 'data/concept_sets/combined_concepts_{}.txt'.format(args.dataset_name.split("_")[0])
        else:
            concept_set = args.concept_set
        concept_activations = linear_explanation.get_clip_feats(clip_name = args.clip_name, dataset_name = args.dataset_name,
                                                concept_set = concept_set,  save_dir = args.activations_dir,
                                                batch_size = args.batch_size, device = args.device)
        with open(concept_set, 'r') as f: 
            concept_text = (f.read()).split('\n')

    elif args.mode == "label":
        concept_activations, concept_text = linear_explanation.get_onehot_labels(args.dataset_name, args.device)

    target_activations = linear_explanation.get_target_acts(target_name = args.target_name, dataset_name = args.dataset_name,
                                                                target_layer = args.target_layer, save_dir = args.activations_dir,
                                                                batch_size = args.batch_size, device = args.device, 
                                                                start_neuron = args.start_neuron, end_neuron = args.end_neuron,
                                                                pool_mode = args.pool_mode)
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    #deleting concepts that are not active enough, or have 0 standard devication
    concept_top5 = torch.mean(torch.topk(concept_activations, dim=0, k=5)[0], dim=0)
    top5_active = (concept_top5 >= 0.5)
    std_active = torch.std(concept_activations, dim=0) >= 1e-5
    active_concepts = top5_active*std_active
    concept_activations = concept_activations[:, active_concepts]
    concept_text = [concept for i, concept in enumerate(concept_text) if active_concepts[i]]
    #print(torch.sum(active_concepts))


    data_utils.save_train_test_split(args.dataset_name)
    train_ids = torch.load("data/data_splits/{}/train_ids.pt".format(args.dataset_name))
    val_ids = torch.load("data/data_splits/{}/val_ids.pt".format(args.dataset_name))

    top_concepts = []
    for i in tqdm(range(math.ceil(target_activations.shape[1]/args.glm_neuron_batch))):
        curr_target = target_activations[:, i*(args.glm_neuron_batch):(i+1)*(args.glm_neuron_batch)]
        train_data, val_data = linear_explanation.get_glm_datasets(concept_activations, curr_target, train_ids, val_ids)

        #train relatively_sparse model
        linear = linear_explanation.train_glm_model(train_data, val_data, args.device, lam=args.glm_lambda)
        vals, curr_top_concepts = torch.sort(linear.weight.detach(), dim=1, descending=True)
        top_concepts.append(curr_top_concepts)
    top_concepts = torch.cat(top_concepts, dim=0)

    result_df = {"layer":[], "unit":[], "val correlation":[]}

    for i in range(args.max_length):
        result_df["weight{}".format(i)] = []
        result_df["concept{}".format(i)] = []
    result_df["bias"] = []
    lengths = []
    end_neuron = args.start_neuron+target_activations.shape[1] #calculate it in case it is args.end_neuron = None
    for n_id, target_neuron in enumerate(range(args.start_neuron, end_neuron)):
        result_df["layer"].append(args.target_layer)
        result_df["unit"].append(target_neuron)
        print("Neuron: {}".format(target_neuron))
        
        train_target = target_activations[train_ids, n_id:n_id+1]
        val_target = target_activations[val_ids, n_id:n_id+1]
        curr_top_concepts = top_concepts[n_id]

        concepts, best_weight, best_bias, best_corr = linear_explanation.greedy_search(concept_activations[train_ids], concept_activations[val_ids],
                                                                                    train_target, val_target, curr_top_concepts, device=args.device,
                                                                                    max_length=args.max_length, tolerance=args.tolerance)
        
        finetune_texts = [concept_text[id] for id in concepts]
        result_df["val correlation"].append(best_corr)
        result_df["bias"].append(best_bias)
        
        new_vals, new_ids = torch.sort(best_weight, descending=True)
        lengths.append(len(new_vals))
        to_print = ""
        for i in range(args.max_length):
            try:
                id = new_ids[i]
                concept, weight = finetune_texts[id], new_vals[i].detach().cpu().numpy()
                result_df["concept{}".format(i)].append(concept)
                result_df["weight{}".format(i)].append(weight)
                to_print += "{:.2f}*{} + ".format(weight, concept)
            except(IndexError):
                result_df["concept{}".format(i)].append(0)
                result_df["weight{}".format(i)].append(0)
        pd_df = pd.DataFrame(result_df)
        pd_df.to_csv("{}/le_{}_{}_{}.csv".format(args.result_dir, args.mode, args.target_name, args.target_layer), index=False)
        to_print += "{:.2f}".format(best_bias)
        print(to_print)
        print("Val Correlation:{:.3f} \n".format(best_corr))

    print("Average val correlation: {:.4f}".format(np.mean(result_df["val correlation"])))
    print("Average explanation length: {:.2f}".format(np.mean(lengths)))


if __name__ == "__main__":
    args = parse_arguments()
    explain_neurons(args)