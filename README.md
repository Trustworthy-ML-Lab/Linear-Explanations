# Linear Explanations for Individual Neurons
This is the official repository for our paper "Linear Explanations for Individual Neurons". Full paper and code will be released soon.

### Method Overview
![Overview](data/images/overview_fig_v4.png)

### Abstract

In recent years many methods have been developed to understand the internal workings of neural networks, often by describing the function of individual neurons in the model. However, these methods typically only focus on explaining the very highest activations of a neuron. In this paper we show this is not sufficient, and that the highest activation range is only responsible for a very small percentage of the neuron's causal effect. In addition, inputs causing lower activations are often very different and can't be reliable predicted by only looking at high activations. We propose that neurons should instead be understood as a linear combination of concepts, and develop an efficient method for producing these linear explanations. In addition, we show how to automatically evaluate description quality using simulation, i.e. predicting neuron activations on unseen inputs in vision setting.

### Example results

![Area Chart](data/images/area_chart.png)

![Qualitative](data/images/nice_example_1.png)

### Simulation results (correlation scoring)

![Correlation score](data/images/correlation_score.png)