# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import argparse
import os 
import time
import numpy as np
from PIL import Image
import math
#
import tensorflow as tf

import i3d

_SAMPLE_VIDEO_FRAMES = 64
_IMAGE_SIZE = 224
_CHECKPOINT_PATHS = {
    'rgb': 'D:\\Data\\Text-to-Clip\\I3D-Feature-Extractor-master\\data\\checkpoints\\rgb_scratch\\model.ckpt',
    'rgb600': 'D:\\Data\\Text-to-Clip\\I3D-Feature-Extractor-master\\data\\checkpoints\\rgb_scratch600\\model.ckpt',
    'rgb_imagenet': 'D:\\Data\\Text-to-Clip\\I3D-Feature-Extractor-master\\data\\checkpoints\\rgb_imagenet\\model.ckpt',
}

def feature_extractor(video_name):

    video_path = os.path.join(VIDEO_DIR, video_name)
    feat_path = os.path.join(OUTPUT_FEAT_DIR, video_name + '.npy')
    n_frames = len([ff for ff in os.listdir(video_path) if ff.endswith('.jpg')])
    batch_frames = 64
    # loading net
    rgb_input = tf.placeholder(tf.float32, shape=(batch_size, batch_frames, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
        net = i3d.InceptionI3d(600, spatial_squeeze=True, final_endpoint='Logits')
        _, end_points = net(rgb_input, is_training=False, dropout_keep_prob=1.0)
    end_feature = end_points['avg_pool3d']
    print(end_feature)
    sess = tf.Session()

    rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable

    saver = tf.train.Saver(var_list=rgb_variable_map,reshape=True)
    saver.restore(sess, _CHECKPOINT_PATHS['rgb600'])

    if os.path.exists(feat_path):
        print('Feature file for video %s already exists.'%video_name)
        return

    print('Total frames: %d'%n_frames)
    
    features = []
    for batch_i in range(math.ceil(n_frames/batch_frames)):
        input_blob = []
        for idx in range(batch_frames):
            idx = (batch_i*batch_frames + idx)%n_frames + 1
            image = Image.open(os.path.join(video_path, '%06d.jpg'%idx))
            image = image.resize((resize_w, resize_h))
            image = np.array(image, dtype='float32')

            image[:, :, :] -= 127.5
            image[:, :, :] /= 127.5
            input_blob.append(image)
        
        input_blob = np.array([input_blob], dtype='float32')

        clip_feature = sess.run(end_feature, feed_dict={rgb_input: input_blob})
        clip_feature = np.reshape(clip_feature, (-1, clip_feature.shape[-1]))
        print(batch_i,clip_feature.shape)
        features.append(clip_feature)

    if len(features)>1:
        features = np.concatenate(features, axis=0)
    else:
        features = features[0]
    features = features[:n_frames//8]
    feat_path = os.path.join(OUTPUT_FEAT_DIR, video_name + '.npy')
    print('Saving features and probs for video: %s ...'%video_name)
    np.save(feat_path, features)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    print('******--------- Extract I3D features ------*******')
    parser.add_argument('-g', '--GPU', type=int, default=0, help='GPU id')
    parser.add_argument('-of', '--OUTPUT_FEAT_DIR', dest='OUTPUT_FEAT_DIR', type=str,
                        default='D:\\Data\\Text-to-Clip\\APP\\video_feature',
                        help='Output feature path')
    parser.add_argument('-vpf', '--VIDEO_PATH_FILE', type=str,
                        default='D:\\Data\\Text-to-Clip\\I3D-Feature-Extractor-master\\charades_sta_videos.txt',
                        help='input video list')
    parser.add_argument('-vd', '--VIDEO_DIR', type=str,
                        default='D:\\Data\\Text-to-Clip\\APP\\video',
                        help='frame directory')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    OUTPUT_FEAT_DIR = params['OUTPUT_FEAT_DIR']
    VIDEO_PATH_FILE = params['VIDEO_PATH_FILE']
    VIDEO_DIR = params['VIDEO_DIR']
    RUN_GPU = params['GPU']

    resize_w = 224
    resize_h = 224
    batch_size = 1

    # set gpu id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(RUN_GPU)

    feature_extractor('00HFP')


