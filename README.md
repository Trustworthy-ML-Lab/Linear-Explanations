# Linear Explanations for Individual Neurons
This is the official repository for our paper ICML 2024 *Linear Explanations for Individual Neurons*. [Arxiv link](https://arxiv.org/abs/2405.06855) Full code will be released soon.

We propose that neurons are best understood as linear combinations of interpretable concepts, and propose an efficient way to generate such explanations: Linear Explanations (LE). In addition we introduce the *simulation* evaluation for vision models.

### Method Overview
![Overview](data/images/overview_fig_v7.png)

### Simulation results (correlation scoring)


| Target model              | Network <br> Dissection | MILAN  | CLIP-Dissect | LE (Label) | LE (SigLIP) |
|---------------------------|------------------------|--------|--------------|------------|-------------|
| ResNet-50 (ImageNet)  | 0.1242                 | 0.0920 | 0.1871       | 0.2924     | **0.3772**  |
| ResNet-18 (Places365) | 0.2038                 | 0.1557 | 0.2208       | 0.3388     | **0.4372**  |
| VGG-16 (CIFAR-100)    | -                      | -      | 0.2298       | 0.4330     | **0.4970**  |
| ViT-B/16 (ImageNet)   | -                      | -      | 0.1722       | 0.3243     | **0.3489**  |
| ViT-L/32 (ImageNet)   | -                      | -      | 0.0549       | 0.1879     | **0.2182**  |

Average correlation scores for different explanation methods in seconds to last layer of the different models.

### Example results

![Qualitative](data/images/nice_example_1_edited.png)

### Cite this work
T. Oikarinen and T.W. Weng, Linear Explanations for Individual Neurons, ICML 2024.

```
@inproceedings{oikarinen2024linear,
  title={Linear Explanations for Individual Neurons},
  author={Oikarinen, Tuomas and Weng, Tsui-Wei},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```