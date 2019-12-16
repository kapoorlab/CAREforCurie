
# coding: utf-8

# In[1]:



from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import os
import glob
from csbdeep.utils.tf import limit_gpu_memory

from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import ProjectionCARE


# In[2]:


basedirLow = '/local/u934/private/v_kapoor/ProjectionTraining/MasterLow/NotsoLow/'
basedirResults = '/local/u934/private/v_kapoor/ProjectionTraining/MasterLow/NetworkProjections'
ModelName = 'DrosophilaDenoisingProjection'
BaseDir = '/local/u934/private/v_kapoor/CurieDeepLearningModels/'


# In[3]:


model = ProjectionCARE(config = None, name = ModelName, basedir = BaseDir)


# In[4]:


Raw_path = os.path.join(basedirLow, '*tif')
Path(basedirResults).mkdir(exist_ok = True)
axes = 'ZYX'
filesRaw = glob.glob(Raw_path)
filesRaw.sort
for fname in filesRaw:
        x = imread(fname)
        print('Saving file' +  basedirResults + '%s_' + os.path.basename(fname))
        restored = model.predict(x, axes, n_tiles = (1, 4, 4)) 
        axes_restored = axes.replace(model.proj_params.axis,'')
        save_tiff_imagej_compatible((basedirResults + '%s_' + os.path.basename(fname)) % model.name, restored, axes_restored)
