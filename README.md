# Keyword Spotting

This repository compares the accuracies and size of different model architecture for the purpose of keyword recognition from audio files. The aim of the project is to minimize model size and FLOPs (floating-point operations per second) so as to make the model feasible to be inferenced on resource constrained device such as IoT (Internet of Things) and Edge devices. The project also explores the use of TensorFlow lite for optimizing the model for inference on target devices. To achieve this the PyTorch model is converted to ONNX (Open Neural Network Exchange) format and then to TensorFlow and TensorFlow lite models. The `src` directory also contains the code for inferencing on any device with a microphone and explores different ways to handle multiple words being spoken in an audio stream. `src/raspberry_inference.py` is the most utility based part of the project and I suggest playing around with it as a starting point to understand the project.

## Training from Source

Run the following command:

```
pip install -r requirements.txt
make
```

## Resources

|paper|link|
|-|-|
|SMALL-FOOTPRINT KEYWORD SPOTTING USING DEEP NEURAL NETWORKS|[paper link](research-papers/42537.pdf)|
|Studying the Effects of Feature Extraction Settings on the Accuracy and Memory Requirements of Neural Networks for Keyword Spotting|[paper link](research-papers/KeywordSpotting_Settings.pdf)|
|EdgeCRNN: an edge‑computing oriented model of acoustic feature enhancement for keyword spotting|[paper link](research-papers/Wei2021_Article_EdgeCRNNAnEdge-computingOrient.pdf)|

## Datasets

|dataset|link|
|-|-|
|Google Speech Commands Dataset|[dataset link](https://arxiv.org/abs/1804.03209)|
