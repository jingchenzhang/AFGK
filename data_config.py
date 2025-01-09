"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

# H36M_ROOT = ''
# MPII_ROOT = ''
# COCO_ROOT = ''
# MPI_INF_3DHP_ROOT = ''
# PW3D_ROOT = ''
BEDLAM_ROOT = ''
AGORA_ROOT = ''

HOME_PATH = '/home/zjc/work_dir/data'
# '/opt/data/private/datasets'
H36M_ROOT = join(HOME_PATH, 'h36m')
LSP_ROOT = join(HOME_PATH, 'lsp_dataset_small')
LSP_ORIGINAL_ROOT = join(HOME_PATH, 'lsp_dataset_original')
LSPET_ROOT = join(HOME_PATH, 'lspet')
MPII_ROOT = join(HOME_PATH, 'mpii')
COCO_ROOT = join(HOME_PATH, 'coco')
MPI_INF_3DHP_ROOT = join(HOME_PATH,'mpi_inf_3dhp')
PW3D_ROOT = join(HOME_PATH, '3DPW')
UPI_S1H_ROOT = join(HOME_PATH, 'human/upi-s1h')
SURREAL_ROOT = join(HOME_PATH, 'human/SURREAL/data')
threeDOH50K_ROOT = join(HOME_PATH, 'human/3DOH50K')

# Output folder to save test/train npz files
DATASET_NPZ_PATH = '/home/zjc/work_dir/code/refit/data/dataset_extras'


# Path to test/train npz files
DATASET_FILES = [ {'3dpw_test': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                   '3dpw_test_sub': join(DATASET_NPZ_PATH, '3dpw_test_sub.npz'),
                   'h36m-p2-mosh': join(DATASET_NPZ_PATH, 'h36m_mosh_valid_p2.npz'),
                  },

                  {'coco': join(DATASET_NPZ_PATH, 'coco-full_train_eft.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_train_eft.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_train.npz'),
                   'h36m': join(DATASET_NPZ_PATH, 'h36m_mosh_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                   'mpi3d_80': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                   'bedlam': join(DATASET_NPZ_PATH, 'bedlam_train.npz'),
                   'agora': join(DATASET_NPZ_PATH, 'agora_train.npz'),
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p2-mosh': H36M_ROOT,
                   'mpii': MPII_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpi3d_80': MPI_INF_3DHP_ROOT,
                   'coco': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                   '3dpw_test': PW3D_ROOT,
                   '3dpw_test_sub': PW3D_ROOT,
                   'bedlam': BEDLAM_ROOT,
                   'agora': AGORA_ROOT
                }


PASCAL_OCCLUDERS = 'data/dataset_extras/pascal_occluders.pkl'
JOINT_REGRESSOR_TRAIN_EXTRA = '/home/zjc/work_dir/code/refit/data/smpl/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/smpl/J_regressor_h36m.npy'
SMPL_MEAN_PARAMS = 'data/smpl/smpl_mean_params.npz'
SMPL_MODEL_DIR = '/home/zjc/work_dir/code/refit/data/smpl'
