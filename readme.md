# Supplementary Code and Data for the Paper *Subsampling and Knowledge Distillation On Adversarial Examples: New Techniques for Deep Learning Based Side Channel Evaluations*

## Overview

This github repository contains the following code:

- A parameterizable implementation of the neural network used in the paper as well as a full implementation of training given the data released in the CHES 2018 CTF contest (AES challenge).
- Two versions of the full challenge solver: one using the neural network, one using the baseline solution.
- Code that takes a set of traces, a set of expanded target key Hamming weights, a ball size and a neural network as input and changes the traces within the l2 radius given into new traces such that the neural network predicts the desired Hamming weight targets.

In addition, the /models directory contains our main neural network model. The baseline solution (lm_baseline.p), the linear classifier trained on adversarial examples for the neural network (lm_adv1.p) and another linear classifier trained on a different set of adversarial examples for the main neural network are available [separately](https://drive.google.com/file/d/1LLBo-FBlgvdRKc3H3eIDEpldh8khnJPr/view?usp=sharing). 

## Requirements

### Dependencies

The code in this repository has been tested with the following dependencies:

- Python (version 3.6.3)
- keras (version 2.1.5)
- tensorflow (version 1.6.0)
- Pycryptosat (Version 5.6.8)
- scikit-learn (version 0.19.1)
- h5py (version 2.7.1)

### System Requirements

In addition to the system requirements imposed by the above dependencies, the system must provide at least about 7 GB of free RAM to be able to replicate the training of the main neural network, since the pre-processing of the trace files loads each training file (6 GB) into working memory in its entirety. Running the pre-trained classifiers and solving the challenge should work on a typical 2020 PC.

## Downloading and Preprocessing the Data

All training datasets of the 2018 CHES challenge are available from Riscure at the [CHES 2018 CTF homepage](https://chesctf.riscure.com/2018/content?show=training). The challenges themselves can also be downloaded there, but do at the time of writing require [registration](https://chesctf.riscure.com/2018/register) and a Google account to download.

Round 4 of the challenge corresponds to Set6 in the paper, i.e. to the unknown device challenge.

In order to replicate training, download the three random-key training files for the AES challenge, pre-process them by running *process_trace_files()* from *utils.py*, and run *training.py* in the same path. The training script will by default train a number of networks of varying depth for 1000 epochs on the same data and then test them on the Set6 challenge data. In order for the last step to work, the Set6 challenge data and the expanded Set6 key must be available in the same path. The expanded Set6 key is contained in the /data subdirectory of this repository. 

## Running the Experiments

- *python3 challenge_solver_nn.py challenge_file model_file* will try to solve the challenge data contained in challenge_file and then print out some statistics on the expected success rate of running the attack with between 1 and 5 traces from the challenge dataset.
- *python3 challenge_solver.py challenge_file model_file* does the same, but in this case model_file should be a linear regression model.
- *occlusion.py* contains a function that allows to compute the impact of zeroizing some part of the trace on the prediction of a neural network model.
- *gradient_descent.pgd* takes as input a set of traces, a neural network, a search radius, and a set of prediction vectors. It then tries to find adversarial inputs within the L2 ball of the given radius around each original trace such that the neural network will output the desired predictions.  
