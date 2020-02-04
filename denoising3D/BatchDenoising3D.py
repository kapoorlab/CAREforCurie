
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
from csbdeep.models import CARE



# In[3]:


basedirLow = '/local/u934/private/v_kapoor/ProjectionTraining/MasterLow/VeryLow/'
basedirResults3D =  '/local/u934/private/v_kapoor/ProjectionTraining/MasterLow/NotsoLow/'
ModelName = 'BorialisS1S2FlorisMidNoiseModel'
BaseDir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/'
Path(basedirResults3D).mkdir(exist_ok = True)



# In[4]:


model = CARE(config = None, name = ModelName, basedir = BaseDir)


# In[6]:


Raw_path = os.path.join(basedirLow, '*tif')


axes = 'ZYX'
smallaxes = 'YX'
filesRaw = glob.glob(Raw_path)

filesRaw.sort
print(len(filesRaw))
for fname in filesRaw:
 
        x = imread(fname)
        print(x.shape)
        print('Saving file' +  basedirResults3D + '%s_' + os.path.basename(fname))
        restored = model.predict(x, axes, n_tiles = (1, 4, 4)) 
        projected = np.max(restored, axis = 0)
        
        save_tiff_imagej_compatible((basedirResults3D + '%s_' + os.path.basename(fname)), restored, axes)



# In[ ]:




    

