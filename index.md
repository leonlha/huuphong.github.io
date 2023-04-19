## Selected projects in pattern recognition, collaborative learning and DL

---

### PSO-Convolutional Neural Networks with Heterogeneous Learning Rate

Development of a distributed collaborative learning framework for image classification by proposing a novel formulation called Dynamics1 and Dynamics2. This formulation incorporates distilled Cuckoo-Smale elements into the particle swarm optimization (PSO) algorithm, using K-Nearest Neighbors in convolutional neural networks (CNNs).

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#) [![](https://img.shields.io/badge/Twitter-white?logo=Twitter)](#) [![](https://img.shields.io/badge/HuggingFace_Transformers-white?logo=huggingface)](#)

[View source code](https://github.com/leonlha/PSO-ConvNet-Dynamics)

### Video Action Recognition Collaborative Learning with Dynamics via PSO-ConvNet Transformer

Extension of the distributed collaborative learning framework to address human action recognition by incorporating two sequence modeling techniques, namely RNN and Transformer. The effectiveness of this approach was validated in the challenging problem of Human Action Recognition (HAR)

Fig.1: A demonstration of the N(n,t) neighborhood, consisting of the positions of four closest particles and particle n itself, is shown. The velocities of the particles are depicted by arrows.
<img src="img/nn_concept.png?raw=true"/>

Fig.2: Rendering End-to-end ConvNet-Transformer Architecture
<img src="img/e2e_cnn_transformer.png?raw=true"/>

Fig.3: Dynamic PSO-ConvNets System Design
<img src="img/dynamic_system.png?raw=true"/>
[View source code](https://github.com/leonlha/Video-Action-Recognition-Collaborative-Learning-with-Dynamics-via-PSO-ConvNet-Transformer)

### Rethinking Recurrent Neural Networks and other Improvements for Image Classification

Proposal of integration of a recurrent Neural Network (RNN) as an additional layer in the design of Convolutional Neural Networks (CNN) for image recognition. The developed end-to-end multi-model ensembles achieve state-of- the-art performance by notably extending the training strategy of the model. The proposed approach results in comparable or even superior results compared to leading models on challenging datasets.

Fig.1: Single E2E-3M Model

<img src="img/sgl_model.png?raw=true"/>

Fig.2: End-to-end Ensembles of Multiple Models: Concept and Design

<img src="img/Multi_Models.png?raw=true"/>
