#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

import os
import glob


from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import ProjectionCARE


# In[2]:


masterdirLow = '/local/u934/private/v_kapoor/ProjectionTraining/MasterLow/NotsoLow'

subdir = next(os.walk(masterdirLow))

ModelName = 'DrosophilaDenoisingProjection'
BaseDir = '/local/u934/private/v_kapoor/CurieDeepLearningModels/'
model = ProjectionCARE(config = None, name = ModelName, basedir = BaseDir)
axes = 'ZYX'



# In[3]:


for x in subdir[1]:
    currentdir = masterdirLow + x
    
    basedirResults2D = currentdir + '/Projections/'
    
    Path(basedirResults2D).mkdir(exist_ok = True)
    
    


    Raw_path = os.path.join(currentdir, '*tif')



    filesRaw = glob.glob(Raw_path)



    for fname in filesRaw:
 
        y = imread(fname)
      
        print('Saving file' +  basedirResults2D + '%s_' + os.path.basename(fname))
        restored = model.predict(y, axes, n_tiles = (1,4,4)) 
        axes_restored = axes.replace(model.proj_params.axis, '')
        save_tiff_imagej_compatible((basedirResults2D + '%s_' + 'Restored' +  os.path.basename(fname)) %  model.name , restored, axes_restored)

        

    
    
   


# In[ ]:




