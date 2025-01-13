class ModelModes(object):
    TRAINING = 'training'
    EVALUATION = "evaluation"

class Setttings(object):
    # exp_dir = '/home/huili/Projects/GAE1/experiments/'
    exp_dir = "/data/epione/user/huili/exp_GAE/"

    # Visdom
    vis_server = 'http://nef-devel2'
    vis_port = 2024

# DatasetDir
MIMIC = {
    "image_root": "/data/epione/user/huili/MIMIC-CXR-JPG-input256/files",
    "image512_root": "/data/epione/user/huili/MIMIC-CXR-JPG-input512/files",
    "df_path": "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-split.csv",
    "chexpert_df_path": "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-split-chexpert.csv",
    "identity_df_path": "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-split-identity.csv",
    "identity_df_valid": "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-valid-pairs.csv",
    "identity_df_test": "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-test-pairs.csv",
    "chexpert_identity_df_path": "/data/epione/user/huili/MIMIC-CXR-JPG-input512/metadata-split-chexpert-identity.csv",
    "latent_root": "/data/epione/user/huili/MIMIC-CXR-JPG-input256/latents_tmp",
    "sementic_root": "/data/epione/user/huili/MIMIC-CXR-JPG-input256/semantics/",
}
# "CheXpert": "/data/epione/user/huili/",
# "Dentex": "",