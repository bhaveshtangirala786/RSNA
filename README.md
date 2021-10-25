# 5th place solution RSNA-MICCAI Brain Tumor Radiogenomic Classification

This an outline of how to reproduce my solution for the [RSNA-MICCAI Brain Tumor Radiogenomic Classification competition](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification)
If you run into any trouble with the setup/code or have any questions please contact me at abhimanyukarshni@gmail.com

**HARDWARE** : Normal kaggle gpu was used for training the model

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.5.1
CUDA 8.0
cuddn 7.1.4.18
nvidia drivers v.384

#DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
# below are the shell commands used in each step, as run from the top level directory
mkdir -p data/stage1/
cd data/stage1/
kaggle competitions download -c <competition name> -f train.csv
kaggle competitions download -c <competition name> -f test_stage_1.csv

mkdir -p data/stage2/
cd ../data/stage1/
kaggle competitions download -c <competition name> -f test_stage_2.csv
cd ..

#DATA PROCESSING
# The train/predict code will also call this script if it has not already been run on the relevant data.
python ./train_code/prepare_data.py --data_dir=data/stage1/ --output_dir=data/stage1_cleaned

#MODEL BUILD: There are three options to produce the solution.
1) very fast prediction
    a) runs in a few minutes
    b) uses precomputed neural network predictions
2) ordinary prediction
    a) expect this to run for 1-2 days
    b) uses binary model files
3) retrain models
    a) expect this to run about a week
    b) trains all models from scratch
    c) follow this with (2) to produce entire solution from scratch

shell command to run each build is below
#1) very fast prediction (overwrites comp_preds/sub1.csv and comp_preds/sub2.csv)
python ./predict_code/calibrate_model.py

#2) ordinary prediction (overwrites predictions in comp_preds directory)
sh ./predict_code/predict_models.sh

#3) retrain models (overwrites models in comp_model directory)
sh ./train_code/train_models.sh
