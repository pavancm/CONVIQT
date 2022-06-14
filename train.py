import numpy as np
import torch
import argparse
import os

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.dataset_loader import video_data_feat

from modules.GRUModel import GRUModel
from modules.nt_xent_multiclass import NT_Xent
from modules.configure_optimizers import configure_optimizers

from model_io import save_model
from modules.sync_batchnorm import convert_model
import time
import datetime
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')
    
def train(args, train_loader_syn, train_loader_ugc, \
          model, criterion, optimizer, scaler, scheduler = None):
    loss_epoch = 0
    model.train()
    
    for step,((syn_i1, syn_i2, dist_label_syn),(ugc_i1, ugc_i2, _)) in \
    enumerate(zip(train_loader_syn, train_loader_ugc)):
        
        #video 1
        syn_i1 = syn_i1.cuda(non_blocking=True)
        ugc_i1 = ugc_i1.cuda(non_blocking=True)
        x_i1 = torch.cat((syn_i1,ugc_i1),dim=0)
        
        #video 2
        syn_i2 = syn_i2.cuda(non_blocking=True)
        ugc_i2 = ugc_i2.cuda(non_blocking=True)
        x_i2 = torch.cat((syn_i2,ugc_i2),dim=0)
        
        x_i1 = x_i1.squeeze(1)
        x_i2 = x_i2.squeeze(1)
        
        # distortion classes
        # synthetic distortion classes
        dist_label = torch.zeros((2*args.batch_size, \
                                  args.clusters+(args.batch_size*args.nodes)))
        dist_label[:args.batch_size,:args.clusters] = dist_label_syn.clone()
        
        # UGC data - each video is unique class
        dist_label[args.batch_size:,args.clusters + (args.nr*args.batch_size) : \
                       args.clusters + ((args.nr+1)*args.batch_size)] = \
            torch.eye(args.batch_size)
        
        dist_label = dist_label.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=True):
            z_i1, z_i2, h_i1, h_i2 = model(x_i1,x_i2)
            loss = criterion(z_i1, z_i2, dist_label)
        
        # update model weights
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()
            
        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))
        
        if args.nr == 0 and step % 5 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Step [{step}/{args.steps}]\t Loss: {loss.item()}\t LR: {round(lr, 5)}")

        if args.nr == 0:
            args.global_step += 1

        loss_epoch += loss.item()
    
    return loss_epoch

def main(gpu, args):
    rank = args.nr * args.gpus + gpu
    
    if args.nodes > 1:
        cur_dir = 'file://' + os.getcwd() + '/sharedfile'
        dist.init_process_group("nccl", init_method=cur_dir,\
                                rank=rank, timeout = datetime.timedelta(seconds=3600),\
                                world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # loader for synthetic distortions data
    train_dataset_syn = video_data_feat(file_path=args.csv_file_syn,\
                                                temporal_len = args.num_frames)

    if args.nodes > 1:
        train_sampler_syn = torch.utils.data.distributed.DistributedSampler(
            train_dataset_syn, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler_syn = None

    train_loader_syn = torch.utils.data.DataLoader(
        train_dataset_syn,
        batch_size=args.batch_size,
        shuffle=(train_sampler_syn is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler_syn,
    )
    
    # loader for authetically distorted data
    train_dataset_ugc = video_data_feat(file_path=args.csv_file_ugc,\
                                                temporal_len = args.num_frames)

    if args.nodes > 1:
        train_sampler_ugc = torch.utils.data.distributed.DistributedSampler(
            train_dataset_ugc, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler_ugc = None

    train_loader_ugc = torch.utils.data.DataLoader(
        train_dataset_ugc,
        batch_size=args.batch_size,
        shuffle=(train_sampler_ugc is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler_ugc,
    )
    
    # initialize ResNet
    args.n_features = 2048
    temporal_model = GRUModel(args.n_features, args.n_features // 2, \
                              args.projection_dim, args.normalize,\
                                  num_layers = 1)
    # initialize model
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        temporal_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    temporal_model = temporal_model.to(args.device)
    
    #sgd optmizer
    args.steps = min(len(train_loader_syn),len(train_loader_ugc))
    args.lr_schedule = 'warmup-anneal'
    args.warmup = 0.1
    args.weight_decay = 1e-4
    args.iters = args.steps*args.end_num
    optimizer, scheduler = configure_optimizers(args, temporal_model, cur_iter=-1)
    
    criterion = NT_Xent(args.batch_size, args.temperature, args.device, args.world_size)
    
    # DDP / DP
    if args.dataparallel:
        temporal_model = convert_model(temporal_model)
        temporal_model = DataParallel(temporal_model)
        
    else:
        if args.nodes > 1:
            temporal_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(temporal_model)
            temporal_model = DDP(temporal_model, device_ids=[gpu]);print(rank);dist.barrier()

    temporal_model = temporal_model.to(args.device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
#    writer = None
    if args.nr == 0:
        print('Training Started')
    
    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)
        
    epoch_losses = []
    args.global_step = 0
    args.current_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        
        loss_epoch = train(args, train_loader_syn, train_loader_ugc, \
          temporal_model, criterion, optimizer, scaler, scheduler)
        
        end = time.time()
        print(np.round(end - start,4))
        
        if args.nr == 0 and epoch % 1 == 0:
            save_model(args, temporal_model, optimizer)
            torch.save({'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict()},\
                        args.model_path + 'optimizer.tar')
        
        if args.nr == 0:
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / args.steps}"
            )
            args.current_epoch += 1
            epoch_losses.append(loss_epoch / args.steps)
            np.save(args.model_path + 'losses.npy',epoch_losses)

    ## end training
    save_model(args, temporal_model, optimizer)

def parse_args():
    parser = argparse.ArgumentParser(description="CONVIQT")
    parser.add_argument('--nodes', type=int, default = 1, help = 'number of nodes', metavar='')
    parser.add_argument('--nr', type=int, default = 0, help = 'rank', metavar='')
    parser.add_argument('--csv_file_syn', type = str, \
                        default = 'csv_files/file_names_syn.csv',\
                            help = 'list of filenames of videos with synthetic distortions')
    parser.add_argument('--csv_file_ugc', type = str, \
                        default = 'csv_files/file_names_ugc.csv',\
                            help = 'list of filenames of UGC videos')
    parser.add_argument('--num_frames', type=tuple, default=16,\
                        help = 'number of frames in video')
    parser.add_argument('--batch_size', type=int, default = 512, \
                        help = 'number of videos in a batch')
    parser.add_argument('--workers', type = int, default = 4, \
                        help = 'number of workers')
    parser.add_argument('--opt', type = str, default = 'sgd',\
                        help = 'optimizer type')
    parser.add_argument('--lr', type = float, default = 0.6*2,\
                        help = 'learning rate')
    parser.add_argument('--model_path', type = str, default = 'checkpoints/',\
                        help = 'folder to save trained models')
    parser.add_argument('--temperature', type = float, default = 0.1,\
                        help = 'temperature parameter')
    parser.add_argument('--clusters', type = int, default = 120,\
                        help = 'number of synthetic distortion classes')
    parser.add_argument('--reload', type = bool, default = False,\
                        help = 'reload trained model')
    parser.add_argument('--normalize', type = bool, default = True,\
                        help = 'normalize encoder output')
    parser.add_argument('--projection_dim', type = int, default = 128,\
                        help = 'dimensions of the output feature from projector')
    parser.add_argument('--dataparallel', type = bool, default = False,\
                        help = 'use dataparallel module of PyTorch')
    parser.add_argument('--start_epoch', type = int, default = 0,\
                        help = 'starting epoch number')
    parser.add_argument('--end_num', type = int, default = 25,\
                        help = 'number to calculate learning rate decay')
    parser.add_argument('--epochs', type = int, default = 10,\
                        help = 'total number of epochs')
    parser.add_argument('--seed', type = int, default = 10,\
                        help = 'random seed')
    args = parser.parse_args()
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.gpus = 1
    args.world_size = args.gpus * args.nodes
    return args

if __name__ == "__main__":
    args = parse_args()
    
    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)