#!/usr/bin/env python3

import sys
sys.path.append('/home/huili/Projects/GAE1/')
from default_config import ModelModes, ModelTypes, Setttings, MIMIC

# For Evaluation
Setttings.exp_dir = '/home/huili/Projects/GAE1/experiments/' # Debug

class identity_args(object):
    """
    Shared config
    """
    name = "Step4: Identity Classification Testing" # Debug

    # Dataset
    dataset = "MIMIC" # ["MIMIC", "CheXpert", "Dentex"]
    csv_path = globals()[dataset]["identity_df_path"]
    image_root = globals()[dataset]["image_root"]
    
    # Reconstruction
    # anony_root = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/reco/"
    # Reconstruction of E-training
    # anony_root = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/reco_e-training/"

    # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.0 lambda_id_ut=[1,1]
    # anony_root = "/data/epione/user/huili/exp_coES/00037-MIMIC_Latent-auto-epoch1000-margin0.0-id1.0-ut1.0/anony"

    # # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.5 lambda_id_ut=[1,1]
    # anony_root = "/data/epione/user/huili/exp_coES/00034-MIMIC_Latent-auto-epoch1000-margin-0.5-id1.0-ut1.0/anony" 

    # # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.7 lambda_id_ut=[1,1]
    # anony_root = "/data/epione/user/huili/exp_coES/00040-MIMIC_Latent-auto-epoch1000-margin-0.7-id1.0-ut1.0/anony"

    # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.0 lambda_id_ut=[1,0] optimization_id.py
    # anony_root = "/data/epione/user/huili/exp_coES/00063-MIMIC_Latent-auto-epoch1000-margin-0.0-id1.0/anony"

    # # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.5 lambda_id_ut=[1,0] optimization_id.py
    # anony_root = "/data/epione/user/huili/exp_coES/00054-MIMIC_Latent-auto-epoch1000-margin-0.5-id1.0/anony"

    # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.7 lambda_id_ut=[1,0] optimization_id.py
    # anony_root = "/data/epione/user/huili/exp_coES/00058-MIMIC_Latent-auto-epoch1000-margin-0.7-id1.0/anony" 

    # Optimization B=1 W lr=0.01 epoch10 lambda_id_ut=[0,1] optimization_ut.py
    # anony_root = "/data/epione/user/huili/exp_coES/00066-MIMIC_Latent-auto-epoch10/anony"

    # Optimization B=1 W lr=0.01 epoch50 lambda_id_ut=[0,1] optimization_ut.py
    # anony_root = "/data/epione/user/huili/exp_coES/00064-MIMIC_Latent-auto-epoch50/anony"

    # # Optimization B=1 W lr=0.01 epoch100 lambda_id_ut=[0,1] optimization_ut.py 
    # anony_root = "/data/epione/user/huili/exp_coES/00049-MIMIC_Latent-auto-epoch100/anony"

    input_nc = 1
    transform = [False, False] # [RandomHorizontalFlip, Rotation]
    shuffle=False # Already when creating pairs in csv
    batch_size = 1 #48
    num_workers = 8

    # Validation
    gpu_ids = 0
    nfolds=10
    pca=0

    # IResNet params
    network = "r50"
    image_dims = (1,256,256)
    # r50 channel_1 transform[hflip, rotate]=[False, False] 
    best_threshold = 1.3580
    model_path = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Identity_r50_2024_02_07_12_20/checkpoints/r50_epoch299_idx105900_2024_02_08_04:32:41.pt"
    
    seed = 2048
    # margin_list = (1.0, 0.5, 0.0)
    embedding_size = 512