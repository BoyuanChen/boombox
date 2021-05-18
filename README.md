# The Boombox: Visual Reconstruction from Acoustic Vibrations

[Boyuan Chen](http://boyuanchen.com/),
[Mia Chiquier](https://www.linkedin.com/in/mia-chiquier-3862b9122),
[Hod Lipson](https://www.hodlipson.com/),
[Carl Vondrick](http://www.cs.columbia.edu/~vondrick/)
<br>
Columbia University
<br>

### [Project Website](https://boombox.cs.columbia.edu/) | [Video](https://www.youtube.com/watch?v=fZn-PIlrxRc) | [Paper](http://arxiv.org/abs/2105.08052)

## Overview
This repo contains the PyTorch implementation for paper "The Boombox: Visual Reconstruction from Acoustic Vibrations".

![teaser](figures/teaser.gif)

## Content

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [About Configs and Logs](#about-configs-and-logs)
- [Training](#training)
- [Evaluation](#evaluation)

## Installation

Our code has been tested on Ubuntu 18.04 with CUDA 11.0. Create a python virtual environment and install the dependencies.

```
virtualenv -p /usr/bin/python3.6 env-boombox
source env-boombox/bin/activate
cd boombox
pip install -r requirements.txt
```

## Data Preparation

Run the following commands to download the dataset (2.0G).

```
cd boombox
wget https://boombox.cs.columbia.edu/dataset/data.zip
unzip data.zip
rm -rf data.zip
```
After this step, you should see a folder named as data, and video and audio data are in `cube`, `small_cuboid` and `large_cuboid` subfolders.

## About Configs and Logs

Before training and evaluation, we first introduce the configuration and logging structure.

1. **Configs:** all the specific parameters used for training and evaluation are indicated as individual config file. Overall, we have two training paradigms: `single-shape` and `multiple-shape`.

    For `single-shape`, we train and evaluate on each shape separately. Their config files are named with their own shape: `cube`, `large_cuboid` and `small_cuboid`. For `multiple-shape`, we mix all the shapes together and perform training and evaluation while the shape is not known a priori. The config file folder is `all`.

    Within each config folder, we have config file for `depth` prediction and `image` prediction. The last digit in each folder refers to the random seed. For example, if you want to train our model with all the shapes mixed to output a RGB image with random seed 3, you should refer the parameters in:
    ```
    configs/all/2d_out_img_3
    ```

2. **Logs:** both the training and evaluation results will be saved in the log folder for each experiment. The last digit in the logs folder indicates the random seed. Inside the logs folder, the structure and contents are:

    ```
    \logs_True_False_False_image_conv2d-encoder-decoder_True_{output_representation}_{seed}
        \lightning_logs
            \checkpoints               [saved checkpoint]
            \version_0                 [training stats]
            \version_1                 [testing stats]
        \pred_visualizations           [predicted and ground-truth images]
    ```

## Training

Both training and evaluation are fast. We provide an example bash script for running our experiments in [run_audio.sh](run_audio.sh). Specifically, to train our model on all shapes that outputs RGB image representations with random seed 1 and GPU 0, run the following command:

```
CUDA_VISIBLE_DEVICES=0 python main.py ./configs/all/2d_out_img_1/config.yaml;
```

## Evaluation

Again, we provide an example bash script for running our experiments in [run_audio.sh](run_audio.sh). Following the above example, to evaluate the trained model, run the following command:

```
CUDA_VISIBLE_DEVICES=0 python eval.py ./configs/all/2d_out_img_1/config.yaml ./logs_True_False_False_image_conv2d-encoder-decoder_True_pixel_1/lightning_logs/checkpoints;
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.