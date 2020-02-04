import sys
sys.path.append("../HelperFunctions")
import numpy as np
import tensorflow as tf
from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches_reduced_target
from selectrawdata import SelectRawData
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"	

# In[2]:



raw_data = SelectRawData.Shuffle_from_folder(
    basepath    = '/local/u934/private/v_kapoor/ProjectionTraining/',
    source_dirs = ['MasterLow/VeryLow', 'MasterLow/NotsoLow'],
    target_dir  = 'GT',
    axes        = 'ZYX'
)


# In[3]:



X, Y, XY_axes = create_patches_reduced_target(
    raw_data            = raw_data,
    patch_size          = (None,128,128),
    n_patches_per_image = 32,
    target_axes         = 'YX',
    reduction_axes      = 'Z',
    save_file           = '/local/u934/private/v_kapoor/ProjectionTraining/DenoisingProjection26-128-128.npz',
)


# In[4]:




