import torch
from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model
from modules.GRUModel import GRUModel
from torchvision import transforms
import numpy as np

import os
import argparse
import pickle
import skvideo.io
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def create_data_loader(image, image_2, batch_size):
    train = torch.utils.data.TensorDataset(image, image_2)
    loader = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        drop_last=True,
        num_workers=12,
        sampler=None,
        shuffle=False
    )
    return loader

def extract_features(args, model, loader):
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

def extract_features_temporal(args, model, loader):
    feat = []
    
    model.eval()
    for step, (batch_im, batch_im_2) in enumerate(loader):
        batch_im = batch_im.type(torch.float32)
        batch_im_2 = batch_im_2.type(torch.float32)
        
        batch_im = batch_im.cuda(non_blocking=True).unsqueeze(0)
        batch_im_2 = batch_im_2.cuda(non_blocking=True).unsqueeze(0)
        
        with torch.no_grad():
            _, _, model_feat, model_feat_2 = model(batch_im, batch_im_2)
        
        feat_ = np.hstack((model_feat.detach().cpu().numpy(),\
                                model_feat_2.detach().cpu().numpy()))
        feat.extend(feat_)
    return np.array(feat)

def main(args):
    # load video
    video = skvideo.io.FFmpegReader(args.video_path)
    T, height, width, C = video.getShape()
    
    #define torch transform for 2 spatial scales
    transform = torch_transform((height, width))
    
    #define arrays to store frames
    frames = torch.zeros((T,3,height,width), dtype=torch.float16)
    frames_2 = torch.zeros((T,3,height// 2,width// 2), dtype=torch.float16)
    
    # read every video frame
    for frame_ind in range(T):
        inp_frame = Image.fromarray(next(video))
        inp_frame, inp_frame_2 = transform(inp_frame)
        frames[frame_ind],frames_2[frame_ind] = \
        inp_frame.type(torch.float16), inp_frame_2.type(torch.float16)
    
    # convert to torch tensors
    loader = create_data_loader(frames, frames_2, args.num_frames)
    
    # load CONTRIQUE Model
    encoder = get_network('resnet50', pretrained=False)
    model = CONTRIQUE_model(args, encoder, 2048)
    model.load_state_dict(torch.load(args.spatial_model_path, map_location=args.device.type))
    model = model.to(args.device)
    
    # extract CONTRIQUE features
    video_feat = extract_features(args, model, loader)
    
    #load CONVIQT model
    temporal_model = GRUModel(c_in = 2048, hidden_size = 1024, \
                              projection_dim = 128, normalize = True,\
                                  num_layers = 1)
    temporal_model.load_state_dict(torch.load(args.temporal_model_path, \
                                              map_location=args.device.type))
    temporal_model = temporal_model.to(args.device)
        
    #extract CONVIQT features
    feat_frames = torch.from_numpy(video_feat[:,:2048])
    feat_frames_2 = torch.from_numpy(video_feat[:,2048:])
    loader = create_data_loader(feat_frames, feat_frames_2, \
                                args.num_frames)
    video_feat = extract_features_temporal(args, temporal_model, loader)
    
    # load regressor model
    regressor = pickle.load(open(args.linear_regressor_path, 'rb'))
    score = regressor.predict(video_feat)[0]
    print(score)
    
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video_path', type=str, \
                        default='sample_videos/30.mp4', \
                        help='Path to video', metavar='')
    parser.add_argument('--spatial_model_path', type=str, \
                        default='models/CONTRIQUE_checkpoint25.tar', \
                        help='Path to trained CONTRIQUE model', metavar='')
    parser.add_argument('--temporal_model_path', type=str, \
                        default='models/CONVIQT_checkpoint10.tar', \
                        help='Path to trained CONVIQT model', metavar='')
    parser.add_argument('--linear_regressor_path', type=str, \
                        default='models/YouTube_UGC.save', \
                        help='Path to trained linear regressor', metavar='')
    parser.add_argument('--num_frames', type=int, \
                        default=16, \
                        help='number of frames fed to GRU', metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)