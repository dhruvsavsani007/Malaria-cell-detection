# Malaria Cell Detection using Custom Images with CNN

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Loss](#loss)
- [Results](#results)


## Introduction

This project focuses on developing a Convolutional Neural Network (CNN) to detect malaria-infected cells from custom microscopic images. The model is trained to classify images as either parasitized or uninfected, providing a valuable tool for early diagnosis and treatment of malaria.

## Installation
To run this project, you'll need to have the following packages installed:

* `pandas`
* `numpy`
* `seaborn`
* `matplotlib`

You can install these packages using `pip`:

```bash
pip install pandas numpy seaborn matplotlib
```

## Dataset

The dataset contains images of parasitized and uninfected cells organized into two folders:
```bash
├── test
│ ├── parasitized
│ ├── uninfected
├── train
├── parasitized
├── uninfected
```

The total dataset consists of 27,558 images.

**Dataset Source:**
This dataset is taken from the official NIH Website: [Malaria Datasets](https://ceb.nlm.nih.gov/repositories/malaria-datasets/)

## Data Preparation

Given the large size of the dataset, it is impractical to load all images into memory simultaneously. Therefore, the data is processed and fed into the model in batches using Keras's `ImageDataGenerator`.

**Image Manipulation:**

Image manipulation techniques such as rotation, resizing, and scaling are applied to make the model more robust to variations in the input images. The `ImageDataGenerator` is used to perform these manipulations automatically.

## Model-architecture
![image](https://github.com/dhruvsavsani007/Malaria-cell-detection/assets/127683401/45e9205e-1df6-4e52-950f-f2504fd39b8a)


## Loss
![image](https://github.com/dhruvsavsani007/Malaria-cell-detection/assets/127683401/941cf7c1-c1cf-4260-827b-6b2f8b8887c1)


## Results
![image](https://github.com/dhruvsavsani007/Malaria-cell-detection/assets/127683401/ca9673c1-0e91-4fa4-9177-d57280a7eb9a)
