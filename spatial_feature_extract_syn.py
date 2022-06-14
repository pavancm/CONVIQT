import torch
from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model
from torchvision import transforms
import numpy as np

import os
import argparse
from PIL import Image
import skvideo.io
import pandas as pd
import scipy.io

class torch_transform:
    def __init__(self, size):
        self.transform1 = transforms.Compose(
            [
                transforms.Resize((size[0],size[1])),
                transforms.ToTensor(),
            ]
        )
        
        self.transform2 = transforms.Compose(
            [
                transforms.Resize((size[0] // 2, size[1] // 2)),
                transforms.ToTensor(),
            ]
        )
    
    def __call__(self, x):
        return self.transform1(x), self.transform2(x)

def rescale_video(vid, min_val, max_val):
    vid = (vid - min_val)/(max_val - min_val + 1e-3)
    return vid

def temporal_filt(vid, filt_choice):
    # choose temporal filter (haar, db2, bior22)
    if filt_choice == 3:
        return vid
    
    filters = ['haar','db2','bior22']
    num_levels = 3
    filt_choice = np.random.choice([0,1,2],1)[0]
    
    #load filter
    filt_path = 'WPT_Filters/' + filters[filt_choice] + '_wpt_' \
        + str(num_levels) + '.mat'
    wfun = scipy.io.loadmat(filt_path)
    wfun = wfun['wfun']
    
    #choose subband
    subband_choice = np.random.choice(np.arange(len(wfun)),1)[0]
    
    #Temporal Filtering
    frame_data = vid.numpy()
    dpt_filt = np.zeros_like(frame_data)
    
    for ch in range(3):
        inp = frame_data[:,ch,:,:].astype(np.float32)
        out = scipy.ndimage.filters.convolve1d(inp,\
                wfun[subband_choice,:],axis=0,mode='constant')
        dpt_filt[:,ch,:,:] = out.astype(np.float16)
    
    min_val, max_val = np.min(dpt_filt), np.max(dpt_filt)
    dpt_filt = torch.from_numpy(dpt_filt)
    dpt_filt = rescale_video(dpt_filt, min_val, max_val)
    return dpt_filt

def create_data_loader(image, image_2, batch_size):
    train = torch.utils.data.TensorDataset(image, image_2)
    loader = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        sampler=None,
        shuffle=False
    )
    return loader

def extract_features(model, loader):
    feat = []
    
    model.eval()
    for step, (batch_im, batch_im_2) in enumerate(loader):
        batch_im = batch_im.type(torch.float32)
        batch_im_2 = batch_im_2.type(torch.float32)
        
        batch_im = batch_im.cuda(non_blocking=True)
        batch_im_2 = batch_im_2.cuda(non_blocking=True)
        with torch.no_grad():
            _,_, _, _, model_feat, model_feat_2, _, _ = model(batch_im, batch_im_2)
            
        feat_ = np.hstack((model_feat.detach().cpu().numpy(),\
                                model_feat_2.detach().cpu().numpy()))
        feat.extend(feat_)
    return np.array(feat)

def main(args):
    # load CONTRIQUE Model
    encoder = get_network('resnet50', pretrained=False)
    model = CONTRIQUE_model(args, encoder, 2048)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device.type))
    model = model.to(args.device)
    
    write_fdr = args.write_fdr
    fls = pd.read_csv(args.csv_file)
    fls = fls.loc[:,'File_names'].tolist()
    
    for step,vid_path in enumerate(fls):
        # extract features for  all temporal transforms
        for temporal in range(4):
            name = vid_path.split('/')[-1]
            sz = int(name.split('sz')[-1].split('_')[0])
            
            content = vid_path.split('/')[-2]
            
            if not os.path.isdir(write_fdr + content):
                os.system('mkdir -p ' + write_fdr + content)
                os.system('chmod -R 777 ' + write_fdr + content)
                
            write_name = write_fdr + content + '/' + \
            name.split('.')[0] + '_temp' + str(temporal) + '.npy'
            
            print(write_name)
            
            if os.path.exists(write_name):
                continue
            
            path = 'training_data/' + vid_path
            vid_raw = skvideo.io.FFmpegReader(path)
            T, H, W, C = vid_raw.getShape()
            
            dist_frames = torch.zeros((T,3,H*sz,W*sz))
            dist_frames = dist_frames.type(torch.float16)
            
            dist_frames_2 = torch.zeros((T,3,H*sz//2,W*sz//2))
            dist_frames_2 = dist_frames_2.type(torch.float16)
            
            transform = torch_transform((H*sz,W*sz))
            for frame_ind in range(T):
                frame = Image.fromarray(next(vid_raw))
                # resize to source spatial resolution
                frame = frame.resize((W*sz, H*sz), Image.LANCZOS)
                frame, frame_2 = transform(frame)
                dist_frames[frame_ind],dist_frames_2[frame_ind] = \
                frame.type(torch.float16), frame_2.type(torch.float16)
            
            # temporal transforms
            dist_frames = temporal_filt(dist_frames, temporal)
            dist_frames_2 = temporal_filt(dist_frames_2, temporal)
            
            loader = create_data_loader(dist_frames, dist_frames_2, args.batch_size)
            video_feat = extract_features(model, loader)
            
            video_feat = video_feat.astype(np.float16)
            np.save(write_name, video_feat)
            vid_raw.close()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, \
                        default='models/CONTRIQUE_checkpoint25.tar', \
                        help='Path to trained CONTRIQUE model', metavar='')
    parser.add_argument('--batch_size', type=int, \
                        default=16, \
                        help='batch size', metavar='')
    parser.add_argument('--csv_file', type=str, \
                        default='csv_files/file_names_syn.csv', \
                        help='path for csv file with filenames', metavar='')
    parser.add_argument('--write_fdr', type=str, \
                        default='training_data/dist_feat/', \
                        help='write folder', metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)