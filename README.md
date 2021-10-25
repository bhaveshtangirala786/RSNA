# 5th place solution RSNA-MICCAI Brain Tumor Radiogenomic Classification

This an outline of how to reproduce my solution for the [RSNA-MICCAI Brain Tumor Radiogenomic Classification competition](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification)

The kaggle notebooks for training can be found [here](https://www.kaggle.com/abhimanyukarshni/rsna-training/notebook) and inference [here](https://www.kaggle.com/abhimanyukarshni/rsna-inference/notebook)

**HARDWARE** : Normal kaggle gpu was used for training the model

# Dataset
The following [kaggle dataset](https://www.kaggle.com/jonathanbesomi/rsna-miccai-png) was used for training 

# Trained Checkpoints
Link to my best model weights [efficientnet-b3](https://drive.google.com/drive/folders/1qSTjlLmP8wrGLD7-qo1hTX5koykH6N_8?usp=sharing)

# Config
Modify the configuration variables in cfg.py before running the code

* TRAIN_PATH : directory with train images
* TEST_PATH : directory with test images
* test_type : 'dcm' if test images are in dicom format, else 'png'
* weights : directory with model weights for predicting

# Usage

* To train the models using a K-fold split, run the following command : python train.py
* To make predictions on new data, run : python predict.py


If you run into any trouble with the setup/code or have any questions please contact me at abhimanyukarshni@gmail.com
