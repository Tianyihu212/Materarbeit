# Demo
![image](https://github.com/Tianyihu212/Materarbeit/blob/main/example.gif)
<br/>
# Masterthesis 
## Content-Based Landmark Retrieval Combining Global and Local Features using Siamese Neural Networks

This paper mainly introduces the use of GLD-v2 dataset for landmark retrieval tasks. This paper uses [Google landmark dataset](https://github.com/cvdfoundation/google-landmark) for image retrieval. The method used is to combine the global and local features of images.<br/>
<br/>
The research points of this paper are divided into 4 points in total:<br/>
<br/>
(1) What is the characteristics of the Landmark Retrieval data set? <br/>
(2) How to better design a Landmark Retrieval system on top of a large-scaled noisy data set? <br/>
(3) Which local feature is suitable for codebook Construction? <br/>
(4) How to incorporate local Features and global Features for landmark re-ranking? <br/>
<br/>
RQ1 I downloaded the data set of the [Google landmark dataset](https://github.com/cvdfoundation/google-landmark), and deeply analyzed the number of noise images in the data set, the gray-scale images in the data set, the number of images in each category, the shooting time of each image, exif information, etc. In-depth analysis. The relevant code is in the [data preprocess folder](https://github.com/Tianyihu212/Materarbeit/tree/main/data%20preprocess). <br/>
<br/>
RQ2 see figure <br/>
![Aaron Swartz](https://github.com/Tianyihu212/Materarbeit/raw/main/framework.png)
```
Overview of the framework pipeline:
1. Ranking uses global feature from query image and document image. 
2. Global feature use Siamese neutral network for training. 
3. I use Efficient Net B0 to extract the global feature and contractive loss function as loss function. 
4. The sift algorithm extracts the local features of the image.
5. Euclidean metric as re ranking methode
6. The final Result Re-ranking outputs the top 10 landmark images.
```
## Contents
* [Installation](#installation)
* [Quickstart](#quickstart)
* [Dataset](#dataset)
* [Models](#models)
* [Train](#train)
* [Evaluate](#evaluate)

## Installation
If you don't have a GPU, you can simply [train](https://github.com/Tianyihu212/Materarbeit/raw/main/train_model_code/latest_ipynb_load_renew.py) the small landmark dataset([Oxford5k](https://paperswithcode.com/dataset/oxford5k) order [Paris6k](https://paperswithcode.com/sota/image-retrieval-on-paris6k) data set) model through colab order jupyter notbook.

## Quickstart

## Dataset
Trian dataset 1,117,843 images. <br/>
<br/>
Test dataset 1,000 images. (sampling from test dataset) <br/>
<br/>
Index dataset 100,000 images. (sampling from training dataset not use im training process)<br/>
<br/>

## Train
In my training process. I will compare the result of 4 experiments from Transfer learning.
<br/>
### [Experiment 1](https://github.com/Tianyihu212/Materarbeit/tree/main/E1)
Pre-trained EfficientNet B0 as feature extractor. <br/>
<br/>
Using pre-trained model in GLD-v2 index dataset(10w) and test dataset(1000) extracted global features from the two datasets. <br/>
<br/>
Evaluate with mAP@100 on retrieval task as baseline. <br/>
<br/>
**Problem** : The GLD-v2 dataset is not suitable for pre-trained trained models im Image_Net dataset. Only the features learned on the IMAGE-NET dataset can be retrieved.
For example: The architecture of the horse in the dataset can be found. <br/>
![Aaron Swartz](https://github.com/Tianyihu212/Materarbeit/blob/main/E1_framwork.png)
<br/>
### [Experiment 2](https://github.com/Tianyihu212/Materarbeit/tree/main/E2)
Transfer learning EfficientNet B0 from ImageNet to GLDV2 dataset with classification top, use conv layer as feature extractor, evaluate with mAP@100 on retrieval task. <br/>
<br/>
Finetured the Efficient model from GLD-v2 dataset. <br/>
<br/>
In this way, the model is more suitable for the GLD-v2 data set. <br/>
<br/>
**Problem**: Train network will overfitting. Because the GLD-v2 dataset has 81313 categories but a few images per category. <br/>
The number of samples in the classification task is very small compared to the number of classes. 
When a network is complex enough then it only needs to simply "remember" the labels of all samples, 
so that a very small training error can be achieved. 
However, the classifier achieved in this way has almost no generalization.
<br/>
![Aaron Swartz](https://github.com/Tianyihu212/Materarbeit/blob/main/E2_framework.png)
<br/>
### [Experiment 3](https://github.com/Tianyihu212/Materarbeit/tree/main/E3)
Siamese network (metric learning) with batch-wise pos/negative mining (all possible pairs within a batch), transfered weights from pre-trianed weights on Image Net, contrastive loss, evaluate with mAP@100 on retrieval task.<br/>
<br/>
**Problem**:No fine-tuning for the structural characteristics of the GLD-v2 data set
<br/>
![Aaron Swartz](https://github.com/Tianyihu212/Materarbeit/blob/main/E3_framwork.png)
<br/>
### [Experiment 4](https://github.com/Tianyihu212/Materarbeit/tree/main/E4)
Siamese network (metric learning) with batch-wise pos/negative mining (all possible pairs within a batch), transfered weights from experiment 2 on GLD-v2 dataset, contrastive loss, evaluate with mAP@100 on retrieval task.<br/>
<br/>
![Aaron Swartz](https://github.com/Tianyihu212/Materarbeit/blob/main/E4_framework.png)



## Evaluate
Evaluate with these index dataset (10w) and test dataset (1000) map@100 results:
```
map@100
E1 : private 11.5% / public 8.08% 
E2 : private 34.93% / public 27.13% 
E3 : private 30.60% / public 23.59% 
E4 : private 36.36% / public 31.16% 
```
