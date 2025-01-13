#!/usr/bin/env python3

import sys
sys.path.append('/home/huili/Projects/GAE1/')
from default_config import Setttings, MIMIC

# For Evaluation
Setttings.exp_dir = '/home/huili/Projects/GAE1/experiments/' # Debug

class utility_args(object):
    """
    Shared config
    """
    name = "Step4: Utility Classification Evaluation ChecXclusion uDense" # Debug
    oui = [False, True, False] # origin_utility_identity

    # Dataset
    dataset = "MIMIC" # ["MIMIC", "CheXpert", "Dentex"]
    csv_path = globals()[dataset]["chexpert_df_path"]
    
    # Reconstruction
    # anony_root = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/reco/"
    # Reconstruction of e-training
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

    input_nc=3
    use_transform = False
    normalize = True
    use_softlabel=False
    num_classes= 4 #14
    train_cols = ['Lung Opacity', 'Atelectasis', 'Pleural Effusion', 'Support Devices']
    # train_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    #               'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
    #               'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']    
    batch_size = 72 #64
    num_workers = 8 #12

    # Evaluation
    gpu_ids = 0
    model = 'DenseNet121'
    # ChecXclusion uDense DenseNet121 pretrained (scheduler SoftLabel-1) 
    warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_02_13_22_24/checkpoints/Step2_MIMIC_Utility_DenseNet121_2024_02_13_22_24_epoch99_idx105900_2024_02_14_16:14.pt"
    
    # DenseNet params
    image_dims = (3,512,512)
    activations='elu'
    last_activation=None # for CELoss / BCEWithLogitsLoss    