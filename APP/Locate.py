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
import torchvision
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import skimage.io as io
from skimage.transform import resize
import i3d
import c3d
from PIL import Image
import math
import sys
import pygame
sys.path.append("D:\\Data\\Text-to-Clip\\SCDM\\grounding\\Charades-STA\\SCDM")
from run_charades_scdm import locate as SCDM_Locate
sys.path.append("D:\\Data\\Text-to-Clip\\SCDM\\grounding\\ActivityNet\\SCDM")
from run_anet_scdm import locate as ExCL_Locate

WEB_MODE = True
IS_LONG = True
LONG_THRESHOLD = 64
if WEB_MODE:
    VIDEO_PATH = "D:\\Data\\Text-to-Clip\\APP\\static\\video"
    FRAME_PATH = "D:\\Data\\Text-to-Clip\\APP\\static\\video_frame"
    FTS_PATH = "D:\\Data\\Text-to-Clip\\APP\\static\\video_feature"
    CLIP_PATH = "D:\\Data\\Text-to-Clip\\APP\\static\\clip"
else:
    VIDEO_PATH = "D:\\Data\\Text-to-Clip\\APP\\static\\video"
    FRAME_PATH = "D:\\Data\\Text-to-Clip\\APP\\static\\video_frame"
    FTS_PATH = "D:\\Data\\Text-to-Clip\\APP\\static\\video_feature"
FPS = 16
I3D_CKPT_PATH = 'D:\\Data\\Text-to-Clip\\I3D-Feature-Extractor-master\\data\\checkpoints\\rgb_scratch600\\model.ckpt'
C3D_CKPT_PATH = 'D:\\Data\\Text-to-Clip\\files\\c3d.pickle'

def video_to_frame(video_path):
    vname = (video_path.split('\\')[-1]).split('.')[0]
    save_path = os.path.join(FRAME_PATH,vname)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        return vname
    # 使用ffmpeg对视频抽帧时，影响输出特征值的还有抽帧质量 参数为 -q:v 不过并不影响最后模型中的使用
    call(["ffmpeg", "-i", video_path,"-r","16","-q:v","5", save_path+"\\%06d.jpg"])
    return vname

def short_video_extraction(vname,n_frames,frame_path,feat_path):

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
    saver.restore(sess, I3D_CKPT_PATH)

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
    print('Saving features for video: %s ...'%vname)
    np.save(feat_path, features)

def long_video_extraction(vname,n_frames,frame_path,feat_path):

    crop_w = 112
    resize_w = 128
    crop_h = 112
    resize_h = 171
    nb_frames = 16

    net = c3d.C3D(487)
    net.load_state_dict(torch.load(C3D_CKPT_PATH))
    EXTRACTED_LAYER = 6
    feature_dim = 4096

    
    total_frames = n_frames
    valid_frames = total_frames/nb_frames * nb_frames
    index_w = np.random.randint(resize_w - crop_w) ## crop
    index_h = np.random.randint(resize_h - crop_h) ## crop
    features = []
    for i in range(int(valid_frames/nb_frames)) :   
        clip = np.array([resize(io.imread(os.path.join(frame_path, '%06d.jpg'%(i+1))), output_shape=(resize_w, resize_h), preserve_range=True) for j in range(i * nb_frames+1, (i+1) * nb_frames+1)])
        clip = clip[:, index_w: index_w+ crop_w, index_h: index_h+ crop_h, :]
        clip = torch.from_numpy(np.float32(clip.transpose(3, 0, 1, 2)))
        clip = Variable(clip)			
        clip = clip.resize(1, 3, nb_frames, crop_w, crop_h)
        _, clip_output = net(clip, EXTRACTED_LAYER) 
        clip_feature  = (clip_output.data).cpu()  
        features.append(clip_feature)
        print(i,clip_feature.cpu().numpy().shape)
    features = torch.cat(features, 0)
    features = features.numpy()
    print('features shape',features.shape)
    print('Saving features for video: %s ...'%vname)       
    np.save(feat_path, features)


def frame_to_fts(vname):

    frame_path = os.path.join(FRAME_PATH, vname)
    feat_path = os.path.join(FTS_PATH, vname + '.npy')
    if os.path.exists(feat_path):
        print('Feature file for video %s already exists.'%vname)
        return
    
    n_frames = len([ff for ff in os.listdir(frame_path) if ff.endswith('.jpg')])
    print('Total frames: %d'%n_frames)

    if IS_LONG:
        long_video_extraction(vname,n_frames,frame_path,feat_path)
    else:
        short_video_extraction(vname,n_frames,frame_path,feat_path)

def text_to_clip(vname,sentence_description):
    

    video_path = os.path.join(VIDEO_PATH,vname)
    vname = video_to_frame(video_path)
    duration = VideoFileClip(video_path).duration
    global LONG_THRESHOLD
    global IS_LONG
    if duration <= LONG_THRESHOLD:
        print("Processing with a short video!")
        IS_LONG = False
    else:
        print("Processing with a long video!")
        IS_LONG = True
    frame_to_fts(vname)
    video_fts_path = os.path.join(FTS_PATH,vname+".npy")
    
    if IS_LONG:
        pred_clip,pred_score = ExCL_Locate(video_fts_path,sentence_description,duration)
    else:
        pred_clip,pred_score = SCDM_Locate(video_fts_path,sentence_description,duration)


    if WEB_MODE:
        clips = []
        for i in range(3):
            clip_dict = {}
            clip_dict['left'] = round(pred_clip[i][0],2)
            clip_dict['right'] = round(pred_clip[i][1],2)
            clip_dict['name'] =  vname+"_"+sentence_description.replace(' ','_')+"_clip%d.mp4"%(i+1)
            clip_dict['score'] = round(pred_score[i],2)
            clip = VideoFileClip(video_path).subclip(pred_clip[i][0],pred_clip[i][1])
            path = os.path.join(CLIP_PATH,clip_dict['name'])
            if not os.path.exists(path):
                clip.write_videofile(path)
            clips.append(clip_dict)
        return clips
    
    else:
        pygame.display.set_caption('predicted clip')
        clip1 = VideoFileClip(video_path).subclip(pred_clip[0][0],pred_clip[0][1])
        clip2 = VideoFileClip(video_path).subclip(pred_clip[1][0],pred_clip[1][1])
        clip3 = VideoFileClip(video_path).subclip(pred_clip[2][0],pred_clip[2][1])
        clip = clips_array([[clip1,clip2,clip3]]).resize(width=1000)
        clip.preview(fps=16, audio=False)
        pygame.quit()

if  __name__ == "__main__":
    text_to_clip("D:\\Data\\Text-to-Clip\\APP\\static\\video\\LNKdVrX_0Fg.mp4","She wraps it around the toy, then tapes it up")