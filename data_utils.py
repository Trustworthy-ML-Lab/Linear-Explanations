import os
import torch
import random
from torchvision import datasets, transforms, models

DATASET_ROOTS = {"imagenet_val": "YOUR PATH"}

def get_target_model(target_name, device):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {architecture}_{dataset}
                i.e. {resnet18_places365, resnet50_imagenet}
                except for resnet18_places this will return a model trained on ImageNet from torchvision
    """
    if target_name == 'resnet18_places365': 
        target_model = models.resnet18(num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    elif "vit" in target_name:
        assert ("_imagenet" in target_name)
        target_name = target_name.replace("_imagenet", "")
        target_name_cap = target_name.replace("vit", "ViT")
        target_name_cap = target_name_cap.replace("_b", "_B")
        target_name_cap = target_name_cap.replace("_l", "_L")
        target_name_cap = target_name_cap.replace("_h", "_H")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    elif "resnet" in target_name:
        assert ("_imagenet" in target_name)
        target_name = target_name.replace("_imagenet", "")
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    elif "cifar" in target_name:
        split = target_name.rindex("_")
        dataset = target_name[split+1:]
        model_name = target_name[:split]
        target_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "{}_{}".format(dataset, model_name), pretrained=True)
        target_model = target_model.to(device)
        preprocess = get_cifar_preprocess()

    target_model.eval()
    return target_model, preprocess

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_cifar_preprocess():
    target_mean = [0.5070, 0.4865, 0.4409]
    target_std = [0.2673, 0.2564, 0.2761]
    preprocess = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                         transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)

    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=preprocess)
        
    return data

def save_train_test_split(dataset_name, save_dir="data/data_splits"):
    save_path = os.path.join(save_dir, dataset_name)
    if os.path.exists(save_path):
        print("Using existing split")
        return
    else:
        os.makedirs(save_path)
    data = get_data(dataset_name)
    n_samples = len(data)
    ids = [i for i in range(n_samples)]
    random.shuffle(ids)
    train_ids = ids[:int(0.7*n_samples)]
    val_ids = ids[int(0.7*n_samples):int(0.8*n_samples)]
    test_ids = ids[int(0.8*n_samples):]
    train_ids = torch.sort(torch.tensor(train_ids))[0]
    val_ids = torch.sort(torch.tensor(val_ids))[0]
    test_ids = torch.sort(torch.tensor(test_ids))[0]

    torch.save(train_ids, os.path.join(save_path, "train_ids.pt"))
    torch.save(val_ids, os.path.join(save_path, "val_ids.pt"))
    torch.save(test_ids, os.path.join(save_path, "test_ids.pt"))
    print("Creating new dataset split")
    return

    
