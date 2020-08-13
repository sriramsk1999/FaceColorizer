## Face Colorizer with CycleGAN

This project is an implementation of the Cycle-consistent adversarial networks (CycleGANs), with results shown on the [Labeled Faces in the Wild dataset](http://vis-www.cs.umass.edu/lfw/). 

A CycleGAN is a modification of a traditional GAN, with the added constraint of *cycle consistency loss*, i.e. training pairs of generators and discriminators such that each generator can transfer from one domain to another and then back.

This project describes the process of colorizing black and white images of faces to colour after having trained on the LFW dataset, with acceptable results.

#### [Model weights link](https://drive.google.com/drive/folders/1RhVaB7IG1yyOS-3fQWa1dCIqj5v3HW1g?usp=sharing)

Download and place the folder `trained/` in the working directory if pretrained weights are to be used.

#### Sample Results from LFW Dataset

![img1](images/a_to_b_3.jpg)

![img2](images/a_to_b_19.jpg)

![img2](images/a_to_b_24.jpg)

#### Colorizing my own face

![ssk](images/ssk.jpg)

### Usage

#### Testing a single image

`python cyclegan.py --test-single path_to_image.jpg`

Uses the pretrained weights in `trained/` to take in a black-and-white image, colorize it and save the result. 

#### Training

Running the `data_loader.py` script will:

1. Download the LFW dataset.
2. Extract and store a subset of the data in `trainA` and `testA`.
3. Create black-and-white versions of the data in `trainB` and `testB`.
4. Read and store images into a `h5` file.

After which, running:

`python cyclegan.py --train`

Will train the network.

### Repo Overview

- `discriminator.py` -> Definition of Discriminator
- `generator.py` -> Definition of Generator
- `model.py` -> Definition of the CycleGAN
- `ops.py` -> Various building blocks for the networks
- `data_loader.py` -> Downloads and prepares data
- `cyclegan.py` -> Training/Inference Script
- `trained/` -> Folder holding weights for pretrained model

### Related: 

[Pix2Pix](https://phillipi.github.io/pix2pix/)

[CycleGAN-Tensorflow](https://github.com/clvrai/CycleGAN-Tensorflow/)
