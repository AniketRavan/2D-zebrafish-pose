# 2D-zebrafish-pose

Here, we use our 3-D physical model to render a synthetic dataset of 2-D projections as seen by the bottom/top camera. A convolutional neural network trained on this synthetic dataset is used to perform pose estimation 2-D video datasets. For 3-D pose estimation, please refer to the following repository:
https://github.com/AniketRavan/3D-fish-pose

## Read the preprint
https://www.biorxiv.org/content/10.1101/2023.01.06.522821v1.full

## Try out the google colab notebook
https://colab.research.google.com/drive/1D20daqPmzXO8bjnBfi6sYFHuGz9nwDom?authuser=1

## Installation

Please open a terminal window on Linux/MacOS and follow the instructions below

Clone the remote git repository containing the source code on your local computer by executing the following command on a terminal 
$ git clone https://github.com/AniketRavan/2D-zebrafish-pose.git

Please make sure that you have anaconda installed on your computer before creating an environment (described below). \
Follow this link for installation instructions (https://docs.anaconda.com/free/anaconda/install/index.html)

After installing anaconda, create an environment named danio_env using anaconda and install all the dependencies for this project using the following command on a terminal \
$ conda create --name danio_env --file conda_environment.yml

Activate the installed environment using \
$ conda activate danio_env

## Getting started

### Generate training dataset

Navigate to the appropriate directory \
$ cd generate_training_data 

Run the python script to generate the training dataset \
$ python runme.py -d <data_folder> -n <n_samples> 

This generates a training dataset of <n_samples> rendered larval images and the corresponding pose in a folder named <data_folder>

### Train a convolutional neural network

Navigate to the appropriate directory \
$ cd train_network_model 

Run the python script to train a network model to perform pose estimation \
$ python runme_four.py -e <n_epochs> -o <output_dir> 

This script trains a network model for <n_epochs> epochs and stores examples of pose prediction outputs for every epoch in <output_dir>

### Perform pose predictions using trained model

Navigate to the appropriate directory \
$ cd pose_prediction 
Run the python script to perform pose predictions and evaluations \

$ python runme_validate_and_evaluate.py


