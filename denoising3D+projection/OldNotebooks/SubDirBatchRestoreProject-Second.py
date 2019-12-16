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
from csbdeep.models import CARE, ProjectionCARE


# In[2]:


masterdirLow = '/data/u934/service_imagerie/v_kapoor/Fl4-2-Var/CAREing_EVvL/'

subdir = next(os.walk(masterdirLow))

RestorationModel = 'BorialisS1S2FlorisMidNoiseModel'
ProjectionModel = 'DrosophilaDenoisingProjection'
BaseDir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/'


RestorationModel = CARE(config = None, name = RestorationModel, basedir = BaseDir)

ProjectionModel = ProjectionCARE(config = None, name = ProjectionModel, basedir = BaseDir)


axes = 'ZYX'



# In[3]:


for x in subdir[1]:
    currentdir = masterdirLow + x
    
    print(currentdir)
    
    basedirResults3D = currentdir + '/3DResults/'

    basedirResults2D = currentdir + '/Projections/'
    
    Path(basedirResults3D).mkdir(exist_ok = True)

    Path(basedirResults2D).mkdir(exist_ok = True)
    
    


    Raw_path = os.path.join(currentdir, '*tif')



    filesRaw = glob.glob(Raw_path)



    for fname in filesRaw:
 
        y = imread(fname)
      
        print('Saving file' +  basedirResults3D + '%s_' + os.path.basename(fname))

        restored = RestorationModel.predict(y, axes, n_tiles = (1,4,8))
        projection = ProjectionModel.predict(restored, axes, n_tiles = (1,4,8)) 
        axes_restored = axes.replace(ProjectionModel.proj_params.axis, '')
        save_tiff_imagej_compatible((basedirResults3D + '%s_' + 'Restored'  + os.path.basename(fname)) % RestorationModel.name, restored, axes)

        save_tiff_imagej_compatible((basedirResults2D + '%s_' + 'Projected' +  os.path.basename(fname)) %  ProjectionModel.name , projection, axes_restored)

        

    
    
   


# In[ ]:




