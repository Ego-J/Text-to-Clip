import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import os, json, h5py, math, pdb, glob, unicodedata, string
import numpy as np

class TACoSDataset(Dataset):
    def __init__(self,opt,mode):
        self.opt = opt
        self.video_fts_path = opt.video_fts_path
        self.wordtoix_path = opt.wordtoix_path
        self.word_fts_path = opt.word_fts_path
        self.video_list = opt.video_list
        if mode == 'train':
            self.data_path = opt.train_data_path
        if mode == 'val':
            self.data_path = opt.val_data_path
        if mode == 'test':
            self.data_path = opt.test_data_path
        self.info = json.load(open(self.data_path))

        self.vnames = []
        self.g_positions = []
        self.sen_emb = []
        self.vlens = []

        List = np.load(self.video_list,encoding="latin1")[mode] # get the train,val or test training video name
        for i in range(len(List)):    
            video_name = str(List[i],encoding="ascii")
            for capidx, caption in enumerate(self.info[video_name]['sentences']):
                if len(caption.split(' ')) < 35:
                    g_left,g_right = self.get_ground_truth_position(self.info[video_name]['timestamps'][capidx])
                    if g_left == -1 or g_right == -1 or g_right-g_left <= 1:
                        continue
                    g_position = [int(g_left),int(g_right)]

                    self.g_positions.append(g_position)
                    self.vnames.append(video_name)
                    self.sen_emb.append(caption)
                    self.vlens.append(int(self.info[video_name]['duration']//29.4))
        
        # print(self.vnames[-1],self.g_positions[-1],len(self.sen_emb[-1]),self.sen_emb[-1],self.vlens[-1])
        # print(len(self.vnames))
        

    def get_ground_truth_position(self, ground_position):
        left_frames = ground_position[0]
        right_frames = ground_position[1]
        left_position = int(left_frames / 29.4)
        right_position = int(right_frames / 29.4)
        if left_position < 0 or right_position < left_position:
            return -1,-1
        else:
            return left_position,right_position
    
    def get_word_embedding(self, current_caption):
        word_emb = np.load(self.word_fts_path,encoding='latin1',allow_pickle=True).tolist()
        wordtoix = np.load(self.wordtoix_path,encoding='latin1',allow_pickle=True).tolist()
        for c in string.punctuation: 
            current_caption = current_caption.replace(c,'')
        current_caption = current_caption.strip()
        if current_caption == '':
            current_caption = '.'
        current_caption_emb = []
        for word in current_caption.lower().split(' '):
            if word in wordtoix:
                current_caption_emb.append(word_emb[wordtoix[word]])
        return current_caption_emb
            

    
    def __getitem__(self, index):
        video_fts = h5py.File(self.video_fts_path,'r')
        return np.array(video_fts[self.vnames[index][:-4]]['c3d_fc6_features']),np.array(self.get_word_embedding(self.sen_emb[index]),dtype=np.float32),self.g_positions[index]   

    def __len__(self):
        return len(self.vnames)


def padding_collate_fn(batch):
    batch = np.array(batch)
    video_fts,sen_emb,g_position = batch[:,0],batch[:,1],batch[:,2]
    video_fts = pad_sequence([torch.from_numpy(x) for x in video_fts],batch_first=True)
    sen_emb = pad_sequence([torch.from_numpy(x) for x in sen_emb],batch_first=True)
    g_position = np.array([x for x in g_position])
    return [video_fts,sen_emb,g_position]