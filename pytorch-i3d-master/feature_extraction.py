import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import csv
import h5py
import os
import cv2
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d
from subprocess import call

def video_to_tensor(pic):
    """
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    Args:  
        pic (numpy.ndarray): Video to be converted to tensor.
    Returns: 
        Tensor: Converted video.
    """
    pic = pic.transpose([3,0,1,2])
    pic =pic[np.newaxis,:,:,:,:]
    print(pic.shape)
    return torch.from_numpy(pic)

def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def video_to_frame(video_path,frame_path,fps):
    vname = (video_path.split('\\')[-1]).split('.')[0]
    save_path = os.path.join(frame_path,vname)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        return vname
    call(["ffmpeg", "-i", video_path,"-r",str(fps), save_path+"\\%06d.jpg"])
    return vname


def frame_to_fts(vname,frame_path,fts_path):
    load_model = 'D:\\Data\\Text-to-Clip\\pytorch-i3d-master\\models\\rgb_imagenet.pt'
    transform = transforms.Compose([videotransforms.CenterCrop(224)])
    i3d = InceptionI3d(400, in_channels=3)
    #i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model,map_location='cpu'))

    num_frames = len(os.listdir(os.path.join(frame_path, vname)))
    imgs = load_rgb_frames(frame_path,vname,1,num_frames)
    imgs = transform(imgs)
    inputs = video_to_tensor(imgs)

    b,c,t,h,w = inputs.shape
    if t > 1600:
        features = []
        for start in range(1, t-56, 1600):
            end = min(t-1, start+1600+56)
            start = max(1, start-48)
            ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cpu(), volatile=True)
            features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
        np.save(os.path.join(fts_path, vname), np.concatenate(features, axis=0))
    else:
        # wrap them in Variable\
        with torch.no_grad():
            inputs = Variable(inputs.cpu())
            features = i3d.extract_features(inputs)
        np.save(os.path.join(fts_path, vname), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())

def video_to_fts(video_path,frame_path,fts_path):
    vname = video_to_frame(video_path,frame_path,fps=16)
    frame_to_fts(vname,frame_path,fts_path)

video_path = "D:\\Data\\Text-to-Clip\\APP\\video\\00HFP.mp4"
frame_path = "D:\\Data\\Text-to-Clip\\APP\\video_frame"
fts_path = "D:\\Data\\Text-to-Clip\\APP\\video_feature"

video_to_fts(video_path,frame_path,fts_path)