# Facial Chirality

---
## Abstract
As a fundamental vision task, facial expression recognition has made substantial progress recently. However, the recognition performance often degrades largely in real-world scenarios due to the lack of robust facial features. In this paper, we propose a simple but effective facial feature learning method that takes the advantage of **facial chirality** to discover the discriminative features for facial expression recognition. Most previous studies implicitly assume that human faces are symmetric. However, our work reveals that the facial asymmetric effect can be a crucial clue. 
Given a face image and its reflection without additional labels, we decouple the reflection-invariant facial features from the input image pair and then demonstrate that the new features with a standard and lightweight learning model (e.g. ResNet-18) are sufficiently robust to outperform the state-of-the-art methods (e.g. SCN in CVPR 2020 and ESRs in AAAI 2020). Our experiments also show the potential of the new features for other facial vision tasks such as expression image retrieval.
![facial chirality](chirality.png)
*An illustration of **facial chirality**. The human face is chiral and its horizontal reflection cannot be superimposed to make the same image, while an achiral object can be perfectly overlapped with its horizontal reflection.*

## Prerequisites
* Install Pytorch==1.7.0
* Install requirements.txt (pip install -r requirements.txt)

## Data Preparation
1. To train our model with RAF-DB dataset, you need to first download the RAF-DB dataset and place it using the structure like:
```
├── raf-basic
│   ├── EmoLabel
│   │   └── list_patition_label.txt
│   └── Image
│       └── aligned
│           ├── train_00001_aligned.jpg
│           ├── ...
│           ├── test_0001_aligned.jpg
│           ├── ...

```
  We only use aligned images for training.

2. Create a directory named "models", download the trained weight of the model for testing from https://drive.google.com/file/d/1bLCY4LKSD7a5GqB_f9-N8pvPYQtbyub0/view?usp=sharing and place the model weight file in the "models" directory.

## Run Testing Using The Provided Model Weight
```
python train.py --test --dataset [path to your RAF-DB datset]
```
## Run Training
```
python train.py --train --dataset [path to your RAF-DB datset]

```

## Run Image Retrieval
```
python train.py --image_retrieval --query_path [path to your query image]

```
