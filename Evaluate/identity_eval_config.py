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
    csv_path = globals()[dataset]["identity_df_test"]
    image_root = globals()[dataset]["image_root"]
    input_nc = 1
    transform = [True, True] # [RandomHorizontalFlip, Rotation]
    shuffle=False # Already when creating pairs in csv
    batch_size = 48
    num_workers = 8

    # Validation
    gpu_ids = 0
    nfolds=10
    pca=0

    # IResNet params
    network = "r50"
    image_dims = (1,256,256)
    # r50 channel_1 transform[hflip, rotate]=[True, True] 
    # best_threshold = 1.1660
    # model_path = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Identity_r50_2024_02_16_18_23/checkpoints/r50_epoch60_idx32269_2024_02_16_23:14.pt"
    # r50 channel_1 transform[hflip, rotate]=[False, False] 
    # best_threshold = 1.3580
    # model_path = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Identity_r50_2024_02_07_12_20/checkpoints/r50_epoch299_idx105900_2024_02_08_04:32:41.pt"
    # best_threshold = 1.3520
    # model_path = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Identity_r50_2024_02_26_20_34/r50_epoch299_idx158700_2024_02_28_07:43:34.pt"
    # r100 channel_1 transform[hflip, rotate]=[True, True] 
    # best_threshold = 1.2320
    # model_path = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Identity_r100_2024_02_16_18_01/checkpoints/r100_epoch299_idx158700_2024_02_18_10:31.pt"
    # r100 channel_1 transform[hflip, rotate]=[False, False] 
    # best_threshold = 1.3290
    # model_path = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Identity_r100_2024_02_21_11_00/checkpoints/r100_epoch299_idx158700_2024_02_23_21:18.pt"
    # best_threshold = 1.5040
    # model_path = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Identity_r100_2024_02_07_12_20/r100_epoch299_idx105900_2024_02_08_15:07.pt"

    seed = 2048
    margin_list = (1.0, 0.5, 0.0)
    embedding_size = 512