from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import zip
from tifffile import imread
from collections import namedtuple
import os
import tqdm
from glob import glob
import numpy as np
import cv2
import collections
from itertools import chain
from keras.preprocessing.image import ImageDataGenerator
from random import sample
from csbdeep.io import save_tiff_imagej_compatible
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
    
 

class SelectRawData(namedtuple('RawData' ,('generator' ,'size' ,'description'))):
    """:func:`collections.namedtuple` with three fields: `generator`, `size`, and `description`.
    Parameters
    ----------
    generator : function
        Function without arguments that returns a generator that yields tuples `(x,y,axes,mask)`,
        where `x` is a source image (e.g., with low SNR) with `y` being the corresponding target image
        (e.g., with high SNR); `mask` can either be `None` or a boolean array that denotes which
        pixels are eligible to extracted in :func:`create_patches`. Note that `x`, `y`, and `mask`
        must all be of type :class:`numpy.ndarray` with the same shape, where the string `axes`
        indicates the order and presence of axes of all three arrays.
    size : int
        Number of tuples that the `generator` will yield.
    description : str
        Textual description of the raw data.
    """

    @staticmethod
    def Shuffle_from_folder(basepath, source_dirs, target_dir, axes='CZYX', pattern='*.tif*', NumTrain = None):
        """Get pairs of corresponding TIFF images read from folders.
        Two images correspond to each other if they have the same file name, but are located in different folders.
        Parameters
        ----------
        basepath : str
            Base folder that contains sub-folders with images.
        source_dirs : list or tuple
            List of folder names relative to `basepath` that contain the source images (e.g., with low SNR).
        target_dir : str
            Folder name relative to `basepath` that contains the target images (e.g., with high SNR).
        axes : str
            Semantics of axes of loaded images (assumed to be the same for all images).
        pattern : str
            Glob-style pattern to match the desired TIFF images.
        Returns
        -------
        RawData
            :obj:`RawData` object, whose `generator` is used to yield all matching TIFF pairs.
            The generator will return a tuple `(x,y,axes,mask)`, where `x` is from
            `source_dirs` and `y` is the corresponding image from the `target_dir`;
            `mask` is set to `None`.
        Raises
        ------
        FileNotFoundError
            If an image found in a `source_dir` does not exist in `target_dir`.
        Example
        --------
        >>> !tree data
        data
        ├── GT
        │   ├── imageA.tif
        │   ├── imageB.tif
        │   └── imageC.tif
        ├── source1
        │   ├── imageA.tif
        │   └── imageB.tif
        └── source2
            ├── imageA.tif
            └── imageC.tif
        >>> data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='YX')
        >>> n_images = data.size
        >>> for source_x, target_y, axes, mask in data.generator():
        ...     pass
        """
        p = Path(basepath)
        pairs = [(f, p/target_dir/f.name) for f in chain(*((p/source_dir).glob(pattern) for source_dir in source_dirs))]
        len(pairs) > 0 or _raise(FileNotFoundError("Didn't find any images."))
        consume(t.exists() or _raise(FileNotFoundError(t)) for s,t in pairs)
        axes = axes_check_and_normalize(axes)
        if NumTrain is None:
         n_images = len(pairs)
        else:
         n_images = NumTrain   
        
        "Shuffle the images in the folder"
        shuffled = sample(pairs, len(pairs))
        
        shuffled_pairs = shuffled[:n_images]
        description = "{p}: target='{o}', sources={s}, axes='{a}', pattern='{pt}'".format(p=basepath, s=list(source_dirs),
                                                                                          o=target_dir, a=axes, pt=pattern)

        def _gen():
            for fx, fy in shuffled_pairs:
                x, y = imread(str(fx)), imread(str(fy))
                len(axes) >= x.ndim or _raise(ValueError())
                yield x, y, axes[-x.ndim:], None

        return SelectRawData(_gen, n_images, description)

    @staticmethod
    def downsample_data(GTdir, Lowdir, SaveGTdir, SaveLowdir,pattern = '*.tif', axes = 'ZYX', downsamplefactor = 0.5, interpolationscheme = cv2.INTER_CUBIC ):
        
        GTimages = sorted(glob(GTdir +  pattern))
        Lowimages = sorted(glob(Lowdir +  pattern))
        
        
        Path(SaveGTdir).mkdir(exist_ok = True)
        Path(SaveLowdir).mkdir(exist_ok = True)
        print(Path(SaveGTdir))
        print(Path(SaveLowdir))
        GT_path = os.path.join(SaveGTdir, '*tif')
        Low_path = os.path.join(SaveLowdir, '*tif')
        
        filesGT = glob.glob(GT_path)
        filesGT.sort
        for fname in filesGT:
          x = imread(fname)
          Name = os.path.basename(os.path.splitext(fname)[0])
          y =   cv2.resize(x, dsize = (x.shape[1] * downsamplefactor, x.shape[0] * downsamplefactor ))
          save_tiff_imagej_compatible((SaveGTdir  + Name) , y,axes)
          print('File saved GT: ' Name, 'size:' y.shape)  
            
        filesLow = glob.glob(Low_path)
        filesLow.sort    
        for fname in filesLow:
          x = imread(fname)
          Name = os.path.basename(os.path.splitext(fname)[0])
          y =   cv2.resize(x, dsize = (x.shape[1] * downsamplefactor, x.shape[0] * downsamplefactor ))
          save_tiff_imagej_compatible((SaveLowdir  + Name) , y,axes)    
          print('File saved Low: ' Name, 'size:' y.shape)    

        
        
    
    @staticmethod
    def from_folder(basepath, source_dirs, target_dir, axes='CZYX', pattern='*.tif*', NumTrain = None, GenerateKeras=False):
        """Get pairs of corresponding TIFF images read from folders.
        Two images correspond to each other if they have the same file name, but are located in different folders.
        Parameters
        ----------
        basepath : str
            Base folder that contains sub-folders with images.
        source_dirs : list or tuple
            List of folder names relative to `basepath` that contain the source images (e.g., with low SNR).
        target_dir : str
            Folder name relative to `basepath` that contains the target images (e.g., with high SNR).
        axes : str
            Semantics of axes of loaded images (must be same for all images).
        pattern : str
            Glob-style pattern to match the desired TIFF images.
        Returns
        -------
        RawData
            :obj:`RawData` object, whose `generator` is used to yield all matching TIFF pairs.
            The generator will return a tuple `(x,y,axes,mask)`, where `x` is from
            `source_dirs` and `y` is the corresponding image from the `target_dir`;
            `mask` is set to `None`.
        Raises
        ------
        FileNotFoundError
            If an image found in `target_dir` does not exist in all `source_dirs`.
        ValueError
            If corresponding images do not have the same size (raised by returned :func:`RawData.generator`).
        Example
        --------
        >>> !tree data
        data
        ├── GT
        │   ├── imageA.tif
        │   └── imageB.tif
        ├── source1
        │   ├── imageA.tif
        │   └── imageB.tif
        └── source2
            ├── imageA.tif
            └── imageB.tif
        >>> data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='YX')
        >>> n_images = data.size
        >>> for source_x, target_y, axes, mask in data.generator():
        ...     pass
        """
        p = Path(basepath)
        image_names = [f.name for f in ( p /target_dir).glob(pattern)]
        
        "Shuffle the images in the folder"
        shuffled = sample(image_names, len(image_names))
        
        if NumTrain is None:
            NumTrain = len(image_names)
        
        "Select some of the shuffled imqges for a smaller training dataset"
        image_names = shuffled[:NumTrain]
        
        NumTrain > 0 or _raise(FileNotFoundError("'target_dir' doesn't exist or didn't find any images in it."))
        consume ((
            ( p / s /n).exists() or _raise(FileNotFoundError( p / s /n))
            for s in source_dirs for n in image_names
        ))
        axes = axes_check_and_normalize(axes)
        xy_name_pairs = [( p / source_dir /n, p/ target_dir / n) for source_dir in source_dirs for n in image_names]
        train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=(0.9, 1.1),
        horizontal_flip=True,
        vertical_flip=True, 
        fill_mode='constant',
        cval=0)
        new_pair=[]
        for fx, fy in xy_name_pairs:
                x, y = imread(str(fx)), imread(str(fy))
                new_pair.append((x,y))
                if GenerateKeras:
                 rankfourX = np.expand_dims(x, axis = -1)
                 rankfourY = np.expand_dims(y, axis = -1)
                 train_generatorX = train_datagen.flow(rankfourX, batch_size= rankfourX.shape[0], seed=1337)
                 train_generatorY=  train_datagen.flow(rankfourY,  batch_size= rankfourY.shape[0], seed=1337)
                 newX = train_generatorX.next()
                 newY = train_generatorY.next()
                 newX = newX[:,:,:,0]
                 newY = newY[:,:,:,0]
                 
                 new_pair.append((newX,newY))
                
        
        print('Using Keras generator for creating a transformed image from each input')
        n_images = len(new_pair)
        description = "{p}: target='{o}', sources={s}, axes='{a}', pattern='{pt}'".format(p=basepath, s=list(source_dirs),
                                                                                        o=target_dir, a=axes, pt=pattern)
      
        
        
        def _newgen():
            for i in range(len(new_pair)):
                x,y = new_pair[i]
                x.shape == y.shape or _raise(ValueError())
                len(axes) >= x.ndim or _raise(ValueError())
                
                yield x, y, axes[-x.ndim:], None
       
                
         
        
        return SelectRawData(_newgen, n_images, description)

def consume(iterator):
    collections.deque(iterator, maxlen=0)


def compose(*funcs):
    return lambda x: reduce(lambda f,g: g(f), funcs, x)

def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes


def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedtuple('Axes',list(allowed))(*[None if axes.find(a) == -1 else axes.find(a) for a in allowed ])

    
def _raise(e):
    raise e

    
def load_json(fpath):
    with open(fpath,'r') as f:
        return json.load(f)


def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        f.write(json.dumps(data,**kwargs))
        
        
        

    
    