# Exploring Inductive Bias in Neural Networks — Senior Thesis 2021 (Danhee Kim, Loyola Marymount University)
## Abstract
Neural networks are powerful tools used for data analysis. Today, the main advantage of neural networks is the ability to outperform nearly every other machine learning algorithm. However, disadvantages of neural networks exist.
Arguably, the biggest problem faced among neural networks today is the "black box" nature behind them: the activity behind neural networks are often unknown to the users and observers. Inductive bias refers to the assumptions given to the model that are helpful when classifying data.
We want to study how neural networks will behave when applying changes in the architecture best suited towards the data we are working with. In this paper, we will be taking a closer look at two different approaches to explore how inductive bias is applied in neural networks. The first approach will be a standard 3 layer fully connected neural network, and the second approach will be a 3 layer neural network with changes made to the structure of the weight matrices. We hope to see better results with a modified architecture.

## Code
This repository contains all the code used to analyze the results done in my senior thesis in mathematics: Exploring Inductive Bias in Neural Networks.
If you would like to run the code in this repository, the desired device must have [Pytorch](http://pytorch.org) installed. Additionally, this code was initially run on CUDA with Google Colab. If CUDA is not available, it can be run on CPU.

## Acknowledgments
This results explored in this experiment would not have been possible without the help of my thesis advisor Dr. Thomas Laurent, Professor of Mathematics at Loyola Marymount University. Thank you!
