import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import os
from tensorflow.python import pywrap_tensorflow


# video_fts_path = "D:\\Data\\Text-to-Clip\\SCDM\\data\\Charades\\charades_i3d_rgb.hdf5"
# video_fts = h5py.File(video_fts_path,'r')

# print(video_fts['00HFP']['i3d_rgb_features'])


# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader('D:\\Data\\Text-to-Clip\\I3D-Feature-Extractor-master\\data\\checkpoints\\rgb_imagenet\\model.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    shape = reader.get_tensor(key).shape
    print(reader.get_tensor(key).shape)

# vname = '00MFE'

# fts = np.load("D:\\Data\\Text-to-Clip\\APP\\video_feature\\"+vname+".npy").squeeze()
# print(fts.shape)
# print(fts)

# data = h5py.File('D:\\Data\\Text-to-Clip\\SCDM\\data\\Charades\\charades_i3d_rgb.hdf5','r')
# fts2 = np.array(data[vname]['i3d_rgb_features'])
# print(fts2.shape)
# print(fts2)
