from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np
import pandas as pd

class video_data_feat(Dataset):
    def __init__(self, file_path, temporal_len = 16, transform = True):
        self.fls = pd.read_csv(file_path)
        self.temporal_len = temporal_len
        self.tranform_toT = transforms.Compose([
                transforms.ToTensor(),
                ])
    
    def __len__(self):
        return len(self.fls)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        vid_path = self.fls.iloc[idx]['File_names']
        
        if 'Kinetics-400' in vid_path:
            vid_path = vid_path.replace('Kinetics-400','/mnt/LIVELAB_NAS/Pavan/kinetics_feat')
            name = vid_path.split('.m')[0]
        elif 'REDS' in vid_path:
            vid_path = vid_path.replace('REDS','')
            vid_path = vid_path.replace('dist','/mnt/LIVELAB_NAS/Pavan/dist_feat')
            name = vid_path.split('.w')[0]
        else:
            vid_path = vid_path.replace('dist','/mnt/LIVELAB_NAS/Pavan/dist_feat')
            name = vid_path.split('.w')[0]
        
        # determine first video characteristics
        div_factor1 = np.random.choice([1,2],1)[0]
        colorspace_choice1 = 4
        temporal_choice1 = 3
        
        feat_path1 = name + '_color' + str(colorspace_choice1) +\
                '_temp' + str(temporal_choice1) + '.npy'
        feat1 = np.load(feat_path1, allow_pickle = True)
        T,D = feat1.shape
        
        #randomly sample a clip of length temporal_len
        T1 = np.random.randint(1, T - 1 - self.temporal_len)
        feat1 = feat1[T1:T1+self.temporal_len,:]
        
        #choose the scale
        if div_factor1 == 1:
            feat1 = feat1[:,:D // 2]
        else:
            feat1 = feat1[:,D//2:]
        
        feat1 = self.tranform_toT(feat1)
        
        # determine second video characteristics
        div_factor2 = 3 - div_factor1
        colorspace_choice2 = 4
        temporal_choice2 = 3
        
        feat_path2 = name + '_color' + str(colorspace_choice2) +\
                '_temp' + str(temporal_choice2) + '.npy'
        feat2 = np.load(feat_path2, allow_pickle=True)
        T,D = feat2.shape
        
        #randomly sample a clip of length temporal_len
        T2 = np.random.randint(1, T - 1 - self.temporal_len)
        feat2 = feat2[T2:T2+self.temporal_len,:]
        
        #choose the scale
        if div_factor2 == 1:
            feat2 = feat2[:,:D // 2]
        else:
            feat2 = feat2[:,D//2:]
        
        feat2 = self.tranform_toT(feat2)
        
        label = self.fls.iloc[idx]['labels']
        label = label[1:-1].split(' ')
        label = np.array([t.replace(',','') for t in label]).astype(np.float32)
        
        return feat1, feat2, label