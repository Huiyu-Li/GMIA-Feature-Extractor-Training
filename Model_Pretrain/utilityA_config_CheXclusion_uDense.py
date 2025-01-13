#!/usr/bin/env python3

import sys
sys.path.append('/home/huili/Projects/GAE1/')
from default_config import Setttings, MIMIC

# Setttings.exp_dir = '/home/huili/Projects/GAE1/experiments/' # Debug

class utility_args(object):
    """
    Shared config
    """
    name = "Step2: Utility Classification Anonymization ChecXclusion uDense" # Debug
    # vis_info = name # [visdom ] # '_' will split window!
    oui = [False, True, False] # origin_utility_identity

    # Dataset
    dataset = "MIMIC" # ["MIMIC", "CheXpert", "Dentex"]
    csv_path = globals()[dataset]["chexpert_df_path"]
    # image_root = globals()[dataset]["image_root"]

    input_nc = 3
    use_transform = True
    normalize = True
    use_softlabel = True
    num_classes= 4 #14
    train_cols = ['Lung Opacity', 'Atelectasis', 'Pleural Effusion', 'Support Devices']
    # train_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    #               'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
    #               'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']    
    batch_size = 64 #64
    num_workers = 8 #12

    # Training
    gpu_ids = 0
    n_epochs = 300
    log_interval = 1000
    save_storage = False

    # DenseNet params
    image_dims = (3,256,256)
    pretrained=True
    activations='elu'
    last_activation=None # for CELoss / BCEWithLogitsLoss 
    
    # Optimizer params
    optim_type = 'adam' #['adam', 'adamw']
    learning_rate = 0.0005 # [1e-4(default: 1e-3), 3e-5(default: 1e-3)]
    weight_decay = 0 #  [1e-6(default: 0), 1e-2(default)]
