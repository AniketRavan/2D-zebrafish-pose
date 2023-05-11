# 2D-zebrafish-pose

Pose estimation of larval zebrafish for 2-D datasets

## Installation

Please open a terminal window on Linux/MacOS and follow the instructions below

Clone git repository:

Clone the remote git repository containing the source code on your local computer by executing the following command on a terminal \ 
$ git clone https://github.com/AniketRavan/2D-zebrafish-pose.git

Create an environment to install all dependencies

Please make sure that you have anaconda installed on your computer before creating an environment. \
Follow this link for installation instructions (https://docs.anaconda.com/free/anaconda/install/index.html)

After installing anaconda, create an environment named danio_env using anaconda and install all the dependencies for this project using the following command on a terminal \
$ conda create --name danio_env --file conda_environment.yml

Activate the installed environment using \
$ conda activate danio_env

## Getting started

### Generate training dataset

Navigate to the appropriate directory \
$ cd generate_training_data \
Run the python script to generate the training dataset \
$ python runme.py -d <data_folder> -n <n_samples> \
This generates a training dataset of <n_samples> rendered larval images and the corresponding pose in a folder named <data_folder>

### Train a convolutional neural network

Navigate to the appropriate directory \
$ cd train_network_model \
Run the python script to train a network model to perform pose estimation \
$ python runme_four.py -e <n_epochs> -o <output_dir> \
This script trains a network model for <n_epochs> epochs and stores examples of pose prediction outputs for every epoch in <output_dir>

### Perform pose predictions using trained model

Navigate to the appropriate directory \
$ cd pose_prediction 
Run the python script to perform pose predictions and evaluations \
$ python runme_validate_and_evaluate.py
