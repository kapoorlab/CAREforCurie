
# coding: utf-8

# In[1]:


from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, ProjectionCARE


# In[2]:



(X,Y), (X_val,Y_val), axes = load_training_data('/local/u934/private/v_kapoor/CurieTrainingDatasets/Drosophilla/DenoisingProjection.npz', validation_split=0.1, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


# In[3]:


# In[4]:


config = Config(axes, n_channel_in, n_channel_out, unet_n_depth=4,train_epochs= 50,train_steps_per_epoch = 400, train_batch_size = 16, train_reduce_lr={'patience': 5, 'factor': 0.5})
print(config)
vars(config)


# In[5]:


model = ProjectionCARE(config, 'DrosophilaDenoisingProjection', basedir='/local/u934/private/v_kapoor/CurieDeepLearningModels')


# In[6]:


history = model.train(X,Y, validation_data=(X_val,Y_val))


# In[7]:



# In[8]:


# In[9]:


model.export_TF()

