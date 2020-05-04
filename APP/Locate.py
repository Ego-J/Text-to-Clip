import numpy as np
import json
import csv
import h5py
import os
import cv2
import numpy as np
from subprocess import call
from moviepy.editor  import VideoFileClip,clips_array
import tensorflow as tf
import i3d
from PIL import Image
import math
import sys
import pygame
sys.path.append("D:\\Data\\Text-to-Clip\\SCDM\\grounding\\Charades-STA\SCDM")
import run_charades_scdm

VIDEO_PATH = "D:\\Data\\Text-to-Clip\\APP\\video"
FRAME_PATH = "D:\\Data\\Text-to-Clip\\APP\\video_frame"
FTS_PATH = "D:\\Data\\Text-to-Clip\\APP\\video_feature"
FPS = 16
CKPT_PATH = 'D:\\Data\\Text-to-Clip\\I3D-Feature-Extractor-master\\data\\checkpoints\\rgb_scratch600\\model.ckpt'

def video_to_frame(video_path):
    vname = (video_path.split('\\')[-1]).split('.')[0]
    save_path = os.path.join(FRAME_PATH,vname)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        duration = len([ff for ff in os.listdir(save_path) if ff.endswith('.jpg')])/FPS
        return vname,duration
    # 使用ffmpeg对视频抽帧时，影响输出特征值的还有抽帧质量 参数为 -q:v 不过并不影响最后模型中的使用
    call(["ffmpeg", "-i", video_path,"-r","16","-q:v","5", save_path+"\\%06d.jpg"])
    duration = len([ff for ff in os.listdir(save_path) if ff.endswith('.jpg')])/FPS
    return vname,duration

def frame_to_fts(vname):

    frame_path = os.path.join(FRAME_PATH, vname)
    feat_path = os.path.join(FTS_PATH, vname + '.npy')
    
    if os.path.exists(feat_path):
        print('Feature file for video %s already exists.'%vname)
        return

    n_frames = len([ff for ff in os.listdir(frame_path) if ff.endswith('.jpg')])
    batch_frames = 64
    # loading net
    rgb_input = tf.placeholder(tf.float32, shape=(1, batch_frames, 224, 224, 3))
    with tf.variable_scope('RGB'):
        net = i3d.InceptionI3d(600, spatial_squeeze=True, final_endpoint='Logits')
        _, end_points = net(rgb_input, is_training=False, dropout_keep_prob=1.0)
    end_feature = end_points['avg_pool3d']
    sess = tf.Session()

    rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable

    saver = tf.train.Saver(var_list=rgb_variable_map,reshape=True)
    saver.restore(sess, CKPT_PATH)

    print('Total frames: %d'%n_frames)
    
    features = []
    for batch_i in range(math.ceil(n_frames/batch_frames)):
        input_blob = []
        for idx in range(batch_frames):
            idx = (batch_i*batch_frames + idx)%n_frames + 1
            image = Image.open(os.path.join(frame_path, '%06d.jpg'%idx))
            image = image.resize((224, 224))
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
    print('Saving features and probs for video: %s ...'%vname)
    np.save(feat_path, features)

def text_to_clip(video_path,sentence_description):
    
    vname , duration = video_to_frame(video_path)
    frame_to_fts(vname)
    video_fts_path = os.path.join(FTS_PATH,vname+".npy")
    pred_clip,pred_score = run_charades_scdm.locate(video_fts_path,sentence_description,duration)

    pygame.display.set_caption('predicted clip')
    clip1 = VideoFileClip(video_path).subclip(pred_clip[0][0],pred_clip[0][1])
    clip2 = VideoFileClip(video_path).subclip(pred_clip[1][0],pred_clip[1][1])
    clip3 = VideoFileClip(video_path).subclip(pred_clip[2][0],pred_clip[2][1])
    clip = clips_array([[clip1,clip2,clip3]]).resize(width=1000)
    clip.preview(fps=16, audio=False)
    pygame.quit()

text_to_clip("D:\\Data\\Text-to-Clip\\APP\\video\\00MFE.mp4","person is drinking")