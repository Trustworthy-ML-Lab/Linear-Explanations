import numpy as np
import argparse
import pandas as pd
import torch
import open_clip

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
    parser.add_argument('--start_neuron', type=int, default=0, help='First neuron')
    parser.add_argument('--end_neuron', type=int, default=None, help='Last neuron, not included. Default is all neurons.')
    parser.add_argument('--calibration_temp', type=float, default=1.1, help='Temperature calibration for loss calculations. Should minimize loss on val data.')
    parser.add_argument('--mode', type=str, default="optim", help="How to determine simulation params, one of: optim, norm")
    return parser.parse_args()


def simulate_ablation_norm(args):
    """
    Evaluates neuron explanations via simulation with ablation scoring.
    Finds params for simulated using statistics (correlation) on validation data similar to OpenAI paper.
    Saves ablation scores in a new column of the explanation df.
    {explanation_path} should be a valid csv file with neuron explanations:
    Requirements:
        - only contains one layer results
        - has a column "unit"
        - has a column "description" OR
        - has columns concept{i} and weight{i}
        - neurons indexed starting from 0
        - CURRENTLY ONLY WORKS for ResNet layer4
    """
    assert "resnet" in args.target_name, "only resnet models currently supported"
    assert args.target_layer == "layer4", "only layer4 currently supported"

    ablation_scores = []
    if args.explanation_path == None:
        explanation_path = utils.EXPLANATION_PATHS[args.explanation_method] + "{}_{}.csv".format(args.target_name, args.target_layer)
    else:
        explanation_path = args.explanation_path
        
    results_df = pd.read_csv(explanation_path)
    target_activations = linear_explanation.get_target_acts(target_name = args.target_name, dataset_name = args.dataset_name,
                                                                   target_layer = args.target_layer, save_dir = args.activations_dir,
                                                                   batch_size = args.batch_size, device = args.device)

    data_utils.save_train_test_split(args.dataset_name)
    val_ids = torch.load("data/data_splits/{}/val_ids.pt".format(args.dataset_name))
    test_ids = torch.load("data/data_splits/{}/test_ids.pt".format(args.dataset_name))

    model, preprocess = data_utils.get_target_model(args.target_name, device=args.device)
    target_data = data_utils.get_data(args.dataset_name, preprocess)

    val_activations = target_activations[val_ids]
    test_activations = target_activations[test_ids]
    #test_outs, val_outs
    test_y = (torch.LongTensor(target_data.targets)[test_ids]).to(args.device)

    #get CLIP image features
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip_name, 
                                                                            pretrained=utils.CN_TO_CHECKPOINT[args.clip_name],
                                                                            device=args.device)
    clip_data = data_utils.get_data(args.dataset_name, clip_preprocess)
    clip_save_name = "{}/{}_{}.pt".format(args.activations_dir, args.dataset_name, args.clip_name)
    utils.save_clip_image_features(clip_model, clip_data, clip_save_name, args.batch_size, args.device)
    clip_image_features = torch.load(clip_save_name, map_location=args.device).float()
    with torch.no_grad():
        clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)

    tokenizer = open_clip.get_tokenizer(args.clip_name)
    a = utils.SIGMOID_PARAMS_IMAGENET_SC[args.clip_name]["a"]
    b = utils.SIGMOID_PARAMS_IMAGENET_SC[args.clip_name]["b"]

    if "sim ablation(norm)" not in results_df.columns:
        results_df["sim ablation(norm)"] = ["-"]*len(results_df)

    if args.end_neuron == None:
        args.end_neuron = max(results_df["unit"])+1
    target_neurons = range(args.start_neuron, args.end_neuron)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():

        orig_outs_test = model.fc(test_activations)/args.calibration_temp
        orig_losses_test = loss_fn(orig_outs_test, test_y)

    for target_neuron in target_neurons:
        neuron_activations = val_activations[:, target_neuron:target_neuron+1]
        with torch.no_grad():
            preds = linear_explanation.predict_activations(results_df, target_neuron, clip_image_features[val_ids],
                                                            clip_model, tokenizer, a, b, args.device).detach()

            sa_mean, sa_std = torch.mean(neuron_activations), torch.std(neuron_activations)
            total_matrix = torch.cat([neuron_activations, preds], dim=1)
            corr_matrix = torch.corrcoef(total_matrix.T)
            correlation = float(corr_matrix[0, 1].cpu().detach())

        with torch.no_grad():
            #baseline
            orig_act = test_activations.detach().clone()
            orig_act[:, target_neuron] =  sa_mean
            outs = model.fc(orig_act)/args.calibration_temp
            baseline_losses = loss_fn(outs, test_y)
            baseline_loss_diff = torch.sum(torch.abs(baseline_losses-orig_losses_test))

            #ablation
            preds = linear_explanation.predict_activations(results_df, target_neuron, clip_image_features[test_ids],
                                                           clip_model, tokenizer, a, b, args.device).detach()
            norm_pred = (preds-torch.mean(preds, dim=0, keepdims=True))/torch.std(preds, dim=0, keepdims=True)
            new_pred = norm_pred * sa_std * correlation + sa_mean

            new_act = torch.cat([orig_act[:, :target_neuron], new_pred, orig_act[:, target_neuron+1:]], dim=1)
            outs = model.fc(new_act)/args.calibration_temp
            ablation_losses = loss_fn(outs, test_y)
            loss_diff = torch.sum(torch.abs(ablation_losses-orig_losses_test))
            ablation_score = (1-loss_diff/baseline_loss_diff).cpu().numpy()
        
        ablation_scores.append(ablation_score)
        results_df.loc[results_df["unit"]==target_neuron, "sim ablation(norm)"] = ablation_score
        print("{}, Neuron:{}, {}, Ablation score: {:.4f}".format(args.target_layer, target_neuron, 
                                                                args.explanation_method, ablation_score.item()))
        results_df.to_csv(explanation_path, index=False)
    print("Explanations from: {}".format(explanation_path))
    return ablation_scores


def simulate_ablation_optim(args):
    """
    Evaluates neuron explanations via simulation with ablation scoring.
    Finds params for simulated value by optimizing on validation set.
    Saves ablation scores in a new column of the explanation df.
    {explanation_path} should be a valid csv file with neuron explanations:
    Requirements:
        - only contains one layer results
        - has a column "unit"
        - has a column "description" OR
        - has columns concept{i} and weight{i}
        - neurons indexed starting from 0
        - CURRENTLY ONLY WORKS for ResNet layer4
    """
    assert "resnet" in args.target_name, "only resnet models currently supported"
    assert args.target_layer == "layer4", "only layer4 currently supported"

    ablation_scores = []
    if args.explanation_path == None:
        explanation_path = utils.EXPLANATION_PATHS[args.explanation_method] + "{}_{}.csv".format(args.target_name, args.target_layer)
    else:
        explanation_path = args.explanation_path
        
    results_df = pd.read_csv(explanation_path)
    target_activations = linear_explanation.get_target_acts(target_name = args.target_name, dataset_name = args.dataset_name,
                                                                   target_layer = args.target_layer, save_dir = args.activations_dir,
                                                                   batch_size = args.batch_size, device = args.device)

    data_utils.save_train_test_split(args.dataset_name)
    val_ids = torch.load("data/data_splits/{}/val_ids.pt".format(args.dataset_name))
    test_ids = torch.load("data/data_splits/{}/test_ids.pt".format(args.dataset_name))

    model, preprocess = data_utils.get_target_model(args.target_name, device=args.device)
    target_data = data_utils.get_data(args.dataset_name, preprocess)

    val_activations = target_activations[val_ids]
    test_activations = target_activations[test_ids]
    #test_outs, val_outs

    val_y = (torch.LongTensor(target_data.targets)[val_ids]).to(args.device)
    test_y = (torch.LongTensor(target_data.targets)[test_ids]).to(args.device)

    #get CLIP image features
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip_name, 
                                                                            pretrained=utils.CN_TO_CHECKPOINT[args.clip_name],
                                                                            device=args.device)
    clip_data = data_utils.get_data(args.dataset_name, clip_preprocess)
    clip_save_name = "{}/{}_{}.pt".format(args.activations_dir, args.dataset_name, args.clip_name)
    utils.save_clip_image_features(clip_model, clip_data, clip_save_name, args.batch_size, args.device)
    clip_image_features = torch.load(clip_save_name, map_location=args.device).float()
    with torch.no_grad():
        clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)

    tokenizer = open_clip.get_tokenizer(args.clip_name)
    a = utils.SIGMOID_PARAMS_IMAGENET_SC[args.clip_name]["a"]
    b = utils.SIGMOID_PARAMS_IMAGENET_SC[args.clip_name]["b"]

    if "sim ablation(optim)" not in results_df.columns:
        results_df["sim ablation(optim)"] = ["-"]*len(results_df)

    if args.end_neuron == None:
        args.end_neuron = max(results_df["unit"])+1
    target_neurons = range(args.start_neuron, args.end_neuron)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        orig_outs_val = model.fc(val_activations)/args.calibration_temp
        orig_losses_val = loss_fn(orig_outs_val, val_y)

        orig_outs_test = model.fc(test_activations)/args.calibration_temp
        orig_losses_test = loss_fn(orig_outs_test, test_y)

    for target_neuron in target_neurons:
        neuron_activations = val_activations[:, target_neuron:target_neuron+1]
        with torch.no_grad():
            preds = linear_explanation.predict_activations(results_df, target_neuron, clip_image_features[val_ids],
                                                            clip_model, tokenizer, a, b, args.device).detach()

        sa_mean = torch.mean(neuron_activations)

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            orig_act = val_activations.detach().clone()
            orig_act[:, target_neuron] =  sa_mean
            outs = model.fc(orig_act)/args.calibration_temp
            baseline_losses = loss_fn(outs, val_y)
            baseline_loss_diff = torch.sum(torch.abs(baseline_losses-orig_losses_val))

        c = torch.tensor(1.0, requires_grad=True, device=args.device)
        d = torch.tensor(0.0, requires_grad=True, device=args.device)
        
        optimizer = torch.optim.Adam(params = [c, d], lr=1e-2)
        orig_act = val_activations.detach().clone()

        #find best params on val set
        for step in range(100):
            optimizer.zero_grad()
            new_pred = c*preds + d
            new_act = torch.cat([orig_act[:, :target_neuron], new_pred, orig_act[:, target_neuron+1:]], dim=1)
            outs = model.fc(new_act)/args.calibration_temp
            ablation_losses = loss_fn(outs, val_y)
            loss_diff = torch.sum(torch.abs(ablation_losses-orig_losses_val))
            loss_diff.backward()
            #print(step, (1-loss_diff/baseline_loss_diff).item(), c.item(), d.item())
            optimizer.step()

        torch.cuda.empty_cache()
        #test data
        with torch.no_grad():
            #baseline
            orig_act = test_activations.detach().clone()
            orig_act[:, target_neuron] =  sa_mean
            outs = model.fc(orig_act)/args.calibration_temp
            baseline_losses = loss_fn(outs, test_y)
            baseline_loss_diff = torch.sum(torch.abs(baseline_losses-orig_losses_test))

            #ablation
            preds = linear_explanation.predict_activations(results_df, target_neuron, clip_image_features[test_ids],
                                                            clip_model, tokenizer, a, b, args.device).detach()
            new_pred = c*preds + d
            new_act = torch.cat([orig_act[:, :target_neuron], new_pred, orig_act[:, target_neuron+1:]], dim=1)
            outs = model.fc(new_act)/args.calibration_temp
            ablation_losses = loss_fn(outs, test_y)
            loss_diff = torch.sum(torch.abs(ablation_losses-orig_losses_test))
            ablation_score = (1-loss_diff/baseline_loss_diff).cpu().numpy()
        
        ablation_scores.append(ablation_score)
        results_df.loc[results_df["unit"]==target_neuron, "sim ablation(optim)"] = ablation_score
        print("{}, Neuron:{}, {}, c:{:.3f}, d:{:.3f}, Ablation score: {:.4f}".format(args.target_layer, target_neuron, 
                                                                                    args.explanation_method, c.item(), d.item(), ablation_score.item()))
        results_df.to_csv(explanation_path, index=False)
    print("Explanations from: {}".format(explanation_path))
    return ablation_scores
    

if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == "optim":
        ablation_scores = simulate_ablation_optim(args)
    elif args.mode == "norm":
        ablation_scores = simulate_ablation_norm(args)
    print("Mean Ablation score:{:.4f}".format(np.mean(ablation_scores)))
    print("Ablation score SEM:{:.4f}".format(np.std(ablation_scores)/np.sqrt(len(ablation_scores))))


