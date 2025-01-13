#!/usr/bin/env python3

import sys
sys.path.append('/home/huili/Projects/GAE1/')
from default_config import Setttings, MIMIC

# Setttings.exp_dir = '/home/huili/Projects/GAE1/experiments/' # Debug

class identity_args(object):
    """
    Shared config
    """
    name = "Step2: Identity Classification Pre-training notrans" # Debug   
    vis_info = name # [visdom ] # '_' will split window!
    oui = [False, False, True] # origin_utility_identity

    # Dataset
    dataset = "MIMIC" # ["MIMIC", "CheXpert", "Dentex"]
    csv_path = globals()[dataset]["identity_df_path"]
    csv_valid_path = globals()[dataset]["identity_df_valid"]
    image_root = globals()[dataset]["image_root"]
    input_nc=1
    use_transform = True
    normalize = False
    num_classes=1599
    batch_size = 48
    num_workers = 12
    # For Testing
    transform = [False, False] # [RandomHorizontalFlip, Rotation]
    shuffle=False # Already when creating pairs in csv
    # nfolds=10
    # pca=0

    # Training
    gpu_ids = 0
    n_epochs = 300
    log_interval = 1000
    save_storage = False

    # IResNet params
    network = "r50"
    image_dims = (1,256,256)
    seed = 2048
    margin_list = (1.0, 0.5, 0.0)
    embedding_size = 512
    warmup_epoch = 0
    save_all_states = True
    resume = False
    model_path = '/data/epione/user/huili/exp_GAE/Step2_MIMIC_Identity_r50_2024_02_15_21_57/checkpoints/r50_epoch237_idx125902_2024_02_16_17:15.pt'
    fp16 = False
    # pretrained=False
    # activations='relu'
    # last_activation=None # for CELoss / BCEWithLogitsLoss 

    # Optimizer params
    # For SGD 
    optim_type = "sgd"
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    # For AdamW
    # optim_type = "adamw"
    # learning_rate = 0.001
    # weight_decay = 0.1

    # Gradient ACC
    gradient_acc = 1

    # Partial FC
    sample_rate = 1.0
    interclass_filtering_threshold = 0

    # For Large Sacle Dataset, such as WebFace42M
    # dali = False 
    # dali_aug = False


"""
Specialized configs
"""
# class identity_args(args):
#     """
#     """