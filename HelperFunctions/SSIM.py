#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:17:32 2019

@author: vkapoor
"""

import sys
import glob
import os
import numpy as np
from tifffile import imread 


from tifffile import imread
import scipy
from scipy import ndimage
from bokeh.models import Label
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.io import export_png, output_notebook
from scipy import stats
from skimage.measure import compare_ssim as ssim
try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path

try:
    import tempfile
    tempfile.TemporaryDirectory
except (ImportError,AttributeError):
    from backports import tempfile
    
    #for X in *.TIF; do mv $X ${X/%.TIF/.tif}; done
   
def ComputeSSIM(filesLow, filesHigh, targetdir):

           SSIM = []
           
           for fname in filesLow:
               
               Low = imread(fname)
               for secondfname in filesHigh:
                   
                   
                   if os.path.basename(secondfname) == os.path.basename(fname):
                          High = imread(secondfname)
                          Structural_Similarity = ssim(Low, High, data_range = Low.max() - Low.min())
                          SSIM.append(Structural_Similarity) 
           MakeHistogram(SSIM, targetdir)               
                          
def MakeHistogram(SSIM, targetdir):

         hist, edges = np.histogram(np.abs(SSIM), density = False, bins = 'auto')
         p = figure(y_axis_label='Counts', x_axis_label='SSIM')
         
         listhist = hist.tolist()
         
         p.quad(top=hist, bottom =0, left=edges[:-1], right=edges[1:],fill_color="#043564", line_color="#643304")
         print(len(listhist))       
         
             
        
         show(p)                 
         #export_png(p, filename=targetdir + 'SSIM'+ '.png')
if __name__ == "__main__":
        Low_dir = '/data/u934/service_imagerie/v_kapoor/Fl4-2-Var/StelaAdditionalProjections/projections/Originalwt_ECadGFPki-hetero_30pre29_bin140x_movN8/LowSNRProjections/'
        High_dir = '/data/u934/service_imagerie/v_kapoor/Fl4-2-Var/StelaAdditionalProjections/projections/Restoredwt_ECadGFPki-hetero_30pre29_bin140x_movN8/HighSNRProjections/'
        save_dir = '/data/u934/service_imagerie/v_kapoor/Fl4-2-Var/StelaAdditionalProjections/projections/'
        
        LowPath = os.path.join(Low_dir, '*.tif')
        HighPath = os.path.join(High_dir, '*.tif')
        
        filesLow = glob.glob(LowPath)
        filesHigh = glob.glob(HighPath)
        ComputeSSIM(filesLow, filesHigh, save_dir)
        
        