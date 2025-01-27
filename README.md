# Floating Waste Detection
This project was created and submitted as my university dissertation. The main idea behing this project was to use a dataset of litter images, taken from every day situations as opposed to in-situ contexts, to train and apply computer vision models to marine waste identification.

## Project Aims
The aims of this project were:
- Train CNN models using general waste images for the purpose of marine waste detection.
- Evaluate the performance of models trained on general data for specific tasks.
- Help propose current, exisiting technological solutions for major environmental issues.

## Methodology
The approach of this solution consisted of two major areas:
- A dataset of waste images in everday settings.
- Evidencing the effectiveness of CNN model's, trained on generalised datasets, applied to the specific task of marine waste detection.

### The Dataset
The dataset of waste images was taken from Kaggle and created by Suman Kunwar and thierrytheg. The dataset consists of 19,762 images of waste data, broken down into 10 distinct classes.
Classes of waste:
- Metal: 1020
- Glass: 3061
- Biological: 997
- Paper: 1680
- Battery: 944
- Trash: 947
- Cardboard: 1825
- Shoes: 1977
- Clothes: 5327
- Plastic: 1984
 
The kaggle dataset can be found here:
Suman Kunwar. (2023). Garbage Dataset [Data set]. Kaggle. (https://doi.org/10.34740/KAGGLE/DS/2814684)

### The Models
5 CNN models were trained on the above dataset via transfer learning. This method of model training was chosen for its convenience, speed, and to demonstrate the approachable nature of the solution through ease of adoption and integration.

The models used in this project were all variations of the CNN model architecture. Architecture choice was again used to demonstrate the effectiveness of existing, well documented, and thoroughly understood computer vision technologies being applied to current sustainability challenges.

Models Used:
- VGG16
- ResNet50
- EfficientNetB7
- Inception_V3
- MobileNetV2
