import argparse
import pandas as pd
import torch
import open_clip
from tqdm import tqdm

import data_utils
import utils
import linear_explanation

def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulation with correlation scoring')
    parser.add_argument('--dataset_name', type=str, default='imagenet_val', help='Dataset name')
    parser.add_argument('--target_name', type=str, default='resnet50_imagenet', help='Target model')
    parser.add_argument('--target_layer', type=str, default='layer4', help='Target layer')
    parser.add_argument('--clip_name', type=str, default='ViT-SO400M-14-SigLIP-384', help='Which CLIP model to use')
    parser.add_argument('--device', type=str, default='cuda', help='whether to use gpu')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for clip and target model')
    parser.add_argument('--activations_dir', type=str, default='saved_activations', help='Directory for neuron activations')
    parser.add_argument('--explanation_method', type=str, default='le_label', help='''Which method results to load. One of {le_label, le_siglip, 
                        net_dissect, milan, clip_dissect}. Not used if explanation_path is provided.''')
    parser.add_argument('--explanation_path', type=str, default=None, help='Path to explanation file. Generates path from other args if not provided')
    parser.add_argument('--pool_mode', type=str, default='avg', help="Activation pooling function for i.e. CNN channel outputs. {avg, max, first}")
    return parser.parse_args()


def simulate_correlations(args):
    """
    Evaluates neuron explanations via simulation with correlation scoring.
    Saves correlation scores in a new column of the explanation df.
    {explanation_path} should be a valid csv file with neuron explanations:
    Requirements:
        - only contains one layer results
        - has a column "unit"
        - has a column "description" OR
        - has columns concept{i} and weight{i}
        - neurons indexed starting from 0
    """
    if args.explanation_path == None:
        explanation_path = utils.EXPLANATION_PATHS[args.explanation_method] + "{}_{}.csv".format(args.target_name, args.target_layer)
    else:
        explanation_path = args.explanation_path
    results_df = pd.read_csv(explanation_path)

    data_utils.save_train_test_split(args.dataset_name)
    test_ids = torch.load("data/data_splits/{}/test_ids.pt".format(args.dataset_name))

    target_activations = linear_explanation.get_target_acts(target_name = args.target_name, dataset_name = args.dataset_name,
                                                            target_layer = args.target_layer, save_dir = args.activations_dir,
                                                            batch_size = args.batch_size, device = args.device, pool_mode=args.pool_mode)
    target_activations = target_activations[test_ids]
    #get CLIP image features
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip_name, 
                                                                           pretrained=utils.CN_TO_CHECKPOINT[args.clip_name],
                                                                           device=args.device)
    clip_data = data_utils.get_data(args.dataset_name, clip_preprocess)
    clip_save_name = "{}/{}_{}.pt".format(args.activations_dir, args.dataset_name, args.clip_name)
    utils.save_clip_image_features(clip_model, clip_data, clip_save_name, args.batch_size, args.device)
    clip_image_features = torch.load(clip_save_name, map_location=args.device).float()[test_ids]
    with torch.no_grad():
        clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)

    tokenizer = open_clip.get_tokenizer(args.clip_name)
    a = utils.SIGMOID_PARAMS_IMAGENET_SC[args.clip_name]["a"]
    b = utils.SIGMOID_PARAMS_IMAGENET_SC[args.clip_name]["b"]
    
    target_neurons = results_df["unit"]
    correlations = []
    for target_neuron in tqdm(target_neurons):
        neuron_activations = target_activations[:, target_neuron:target_neuron+1]
        preds = linear_explanation.predict_activations(results_df, target_neuron, clip_image_features,
                                                        clip_model, tokenizer, a, b, args.device)
        
        total_matrix = torch.cat([neuron_activations, preds], dim=1)
        corr_matrix = torch.corrcoef(total_matrix.T)
        correlations.append(float(corr_matrix[0, 1].cpu().detach()))
    print("Explanations from: {}".format(explanation_path))
    print("Average correlation: {:.4f} +- {:.4f}".format(torch.mean(torch.tensor(correlations)),
                                                        torch.std(torch.tensor(correlations))/(len(correlations))**0.5))
    results_df["sim correlation"] = correlations
    results_df.to_csv(explanation_path, index=False)

if __name__ == "__main__":
    args = parse_arguments()
    simulate_correlations(args)


