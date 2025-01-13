'''
Privacy metrics in the identity semantic space
'''
import os, time, datetime
import pandas as pd
import torch
import numpy as np
import statistics

# pip3 install faiss
# Custom modules
import sys
sys.path.append('/home/huili/Projects/GAE1/')
from default_config import Setttings
from src.helpers import utils

from privacy_metrics.avatars_are_k_hit import avatars_are_k_hit
from privacy_metrics.hidden_rate import hidden_rate
from privacy_metrics.local_cloaking import get_local_cloaking
from privacy_metrics.dcr_nndr import get_dcr, get_nndr
from privacy_metrics.record_to_avatar_distance import record_to_avatar_distance

csv_path = "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-split.csv"

# StyleGAN2 E-S-D Co-training 
latent_root = "/data/epione/user/huili/MIMIC-CXR-JPG-input256/semantics/" 

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

Setttings.exp_dir = '/home/huili/Projects/GAE1/experiments/' # Debug
start_time = time.time()
# Set ouput directories
special_info = f'Privacy Metrics'; directories = Setttings.exp_dir
time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
name = '{}_{}_{}'.format('Step5', special_info, time_signature)
snapshot = os.path.join(directories, name)
utils.makedirs(snapshot)
logger = utils.logger_setup(logpath=os.path.join(snapshot, 'logs'), filepath=os.path.abspath(__file__))

# load data from csv
df = pd.read_csv(csv_path)
# read train-valid-test split
df = df[df['split']=='test']
# Test        
# df = df[:10]

num_images = len(df); embeddings_shape=512
S_real_id_arr = np.zeros((num_images, embeddings_shape)) 
S_anony_id_arr = np.zeros((num_images, embeddings_shape)) 
for idx in range(num_images):
    row = df.iloc[idx]
    fname = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}'
    latent_path = f'{latent_root}/{fname}/'
    anony_path = f'{anony_root}/{fname}/'

    # Read Latents
    S_real_id = torch.load(os.path.join(latent_path, 'S_id.pt')) # [512]
    S_anony_id = torch.load(os.path.join(anony_path, 'S_id.pt'))
    # S_anony_id = S_real_id
    # print(f'S_real_id:{S_real_id.shape}, S_anony_id:{S_anony_id.shape}')
    # Collect Latents
    S_real_id_arr[idx] = S_real_id
    S_anony_id_arr[idx] = S_anony_id

S_real_id_df = pd.DataFrame(S_real_id_arr)
S_anony_id_df = pd.DataFrame(S_anony_id_arr)

are_first_hit = avatars_are_k_hit(S_real_id_df, S_anony_id_df,
                                  distance_metric="minkowski", k=1)
# Hidden rate
hidden_rate = hidden_rate(are_first_hit)
# Local cloaking
local_cloaking = get_local_cloaking(S_real_id_df, S_anony_id_df)

dcr_list = get_dcr(S_real_id_df, S_anony_id_df)
nndr_list = get_nndr(S_real_id_df, S_anony_id_df)
distance_list = record_to_avatar_distance(S_real_id_df, S_anony_id_df)
# Calculate the median
dcr = statistics.median(dcr_list)
nndr = statistics.median(nndr_list)
distance = statistics.median(distance_list)

logger.info(f'Hidden Rate (>90%): {hidden_rate:.3f}')
logger.info(f'Local Cloaking (>5): {local_cloaking}')
logger.info(f'DCR (>0.2*): {dcr:.3f}')   
logger.info(f'NNDR (>0.3*, [0, 1]):{nndr:.3f}')
logger.info(f'Distance: {distance:.3f}')
logger.info('Time {:.3f} min'.format((time.time() - start_time) / 60))
