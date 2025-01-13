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
    image_root = globals()[dataset]["image_root"]
    input_nc=3
    use_transform = False
    normalize = True
    use_softlabel=False
    num_classes= 4 #14
    train_cols = ['Lung Opacity', 'Atelectasis', 'Pleural Effusion', 'Support Devices']
    # train_cols = ['Cardiomegaly', 'Lung Opacity', 'Edema', 'Atelectasis', 'Pleural Effusion']
    # train_cols = ['Cardiomegaly', 'Lung Opacity', 'Edema', 'Atelectasis', 'Pleural Effusion', 'Support Devices']
    # train_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                #   'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
                #   'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']    
    batch_size = 48 #64
    num_workers = 8 #12

    # Evaluation
    gpu_ids = 0
    model = 'DenseNet121'
    warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_02_13_22_24/checkpoints/Step2_MIMIC_Utility_DenseNet121_2024_02_13_22_24_epoch99_idx105900_2024_02_14_16:14.pt"
    # 14 classes
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_18_23_52/checkpoints/epoch99_idx79400_2024_07_19_05:49.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_18_23_52/checkpoints/epoch199_idx158800_2024_07_19_11:46.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_18_23_52/checkpoints/epoch299_idx238200_2024_07_19_17:44.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_18_23_52/checkpoints/epoch399_idx317600_2024_07_19_23:41.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_18_23_52/checkpoints/epoch499_idx397000_2024_07_20_05:40.pt"
    # 5 classes
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_19_00_00/checkpoints/epoch99_idx79400_2024_07_19_05:58.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_19_00_00/checkpoints/epoch199_idx158800_2024_07_19_11:55.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_19_00_00/checkpoints/epoch299_idx238200_2024_07_19_17:52.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_19_00_00/checkpoints/epoch399_idx317600_2024_07_19_23:50.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_19_00_00/checkpoints/epoch499_idx397000_2024_07_20_05:50.pt"
    # 6 classes
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_22_14_18/checkpoints/epoch99_idx79400_2024_07_22_20:04.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_22_14_18/checkpoints/epoch199_idx158800_2024_07_23_01:50.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_22_14_18/checkpoints/epoch299_idx238200_2024_07_23_07:36.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_22_14_18/checkpoints/epoch399_idx317600_2024_07_23_13:21.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_22_14_18/checkpoints/epoch499_idx397000_2024_07_23_19:08.pt"
    
    # Reconstruction
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_29_15_35/checkpoints/epoch99_idx79400_2024_07_29_21:19.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_29_15_35/checkpoints/epoch199_idx158800_2024_07_30_03:02.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_29_15_35/checkpoints/epoch299_idx238200_2024_07_30_08:46.pt"

    # Reconstruction of e-training
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_10_07_23_17/checkpoints/epoch100_idx80194_2024_10_08_17:00.pt"

    # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.0 lambda_id_ut=[1,1]
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_15_17_29/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch99_idx79400_2024_07_16_01:51.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_15_17_29/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch199_idx158800_2024_07_16_10:13.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_15_17_29/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch299_idx238200_2024_07_16_18:35.pt"

    # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.5 lambda_id_ut=[1,1]
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_15_17_29/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch99_idx79400_2024_07_16_01:52.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_15_17_29/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch199_idx158800_2024_07_16_10:15.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_15_17_29/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch299_idx238200_2024_07_16_18:38.pt"

    # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.7 lambda_id_ut=[1,1]
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_17_11_11/checkpoints/epoch99_idx79400_2024_07_17_19:35.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_17_11_11/checkpoints/epoch199_idx158800_2024_07_18_04:10.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_17_11_11/checkpoints/epoch299_idx238200_2024_07_18_12:32.pt"

    # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.0 lambda_id_ut=[1,0] optimization_id.py
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_25_22_49/checkpoints/epoch99_idx79400_2024_07_26_04:33.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_25_22_49/checkpoints/epoch199_idx158800_2024_07_26_10:17.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_25_22_49/checkpoints/epoch299_idx238200_2024_07_26_16:01.pt"

    # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.5 lambda_id_ut=[1,0] optimization_id.py
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_16_09_50/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch99_idx79400_2024_07_16_18:12.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_16_09_50/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch199_idx158800_2024_07_17_02:33.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_16_09_50/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch299_idx238200_2024_07_17_10:55.pt"

    # Optimization B=1 W lr=0.01 ealry_stop=1e-8 id_margin=-0.7 lambda_id_ut=[1,0] optimization_id.py
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_18_11_15/checkpoints/epoch99_idx79400_2024_07_18_19:37.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_18_11_15/checkpoints/epoch199_idx158800_2024_07_19_03:59.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_18_11_15/checkpoints/epoch299_idx238200_2024_07_19_12:21.pt"

    # Optimization B=1 W lr=0.01 epoch10 lambda_id_ut=[0,1] optimization_ut.py
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_25_22_52/checkpoints/epoch99_idx79400_2024_07_26_04:35.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_25_22_52/checkpoints/epoch199_idx158800_2024_07_26_10:17.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_25_22_52/checkpoints/epoch299_idx238200_2024_07_26_16:00.pt"

    # Optimization B=1 W lr=0.01 epoch50 lambda_id_ut=[0,1] optimization_ut.py
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_26_17_32/checkpoints/epoch99_idx79400_2024_07_26_23:15.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_26_17_32/checkpoints/epoch199_idx158800_2024_07_27_04:58.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_26_17_32/checkpoints/epoch299_idx238200_2024_07_27_10:42.pt"

    # Optimization B=1 W lr=0.01 epoch100 lambda_id_ut=[0,1] optimization_ut.py
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_15_17_31/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch99_idx79400_2024_07_16_01:53.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_15_17_31/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch199_idx158800_2024_07_16_10:15.pt"
    # warmstart_ckpt = "/data/epione/user/huili/exp_GAE/Step2_MIMIC_Utility_DenseNet121_2024_07_15_17_31/checkpoints/Step2: Utility Classification Anonymization ChecXclusion uDense_epoch299_idx238200_2024_07_16_18:37.pt"

    # DenseNet params
    image_dims = (3,512,512)
    activations='elu'
    last_activation=None # for CELoss / BCEWithLogitsLoss    