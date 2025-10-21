import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import random
import shutil
import time
import warnings
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import os
import wandb
import timm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from typing import List
from tqdm import tqdm
from enum import Enum
from model_zoo import CSR,CustomDataset
from torch.utils.data import DataLoader
from utils import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--pretrained_emb', metavar='DIR', nargs='?',
                    default='./imgenet1k_train_emb.npy',
                    help='path to pre-trained embeddings dataset (default: imagenet)')
parser.add_argument('--dummy', default=False, type=bool, help='whether to use dumy datasets')
parser.add_argument('--model_name', default='resnet50d.ra4_e3600_r224_in1k', help='timm model name')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:29503', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1227, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--use_ddp', default=False, action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--gpu', default=1, type=int, help='GPU id to use.')

# training parameter
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024 * 4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=4e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


# CSR parameters
parser.add_argument('--csr', default=True, dest='train_csr',
                    action='store_true', help='whether to train CSR model')
parser.add_argument('--use_CL', default=True,
                    action='store_true', help='whether to use contrastiv loss')
parser.add_argument('--use_label_CL',
                    action='store_true', help='whether to use label-based contrastive loss')
parser.add_argument('--topk', default=8, type=int, dest='topk',
                    help='the number of final topk (after annealing)')
parser.add_argument('--initial_topk', default=None, type=int, dest='initial_topk',
                    help='the number of initial topk (before annealing). If None, will be set to input_dim * 10')
parser.add_argument('--k_decay_ratio', default=0.7, type=float, dest='k_decay_ratio',
                    help='ratio of total training steps when k reaches topk (default: 0.7 means k reaches topk at 70% of training)')
parser.add_argument('--auxk', default=512, type=int, dest='auxk',
                    help='the number of auxk')
parser.add_argument('--cl_coef', default=0.1, type=float,help='cl_coef')
parser.add_argument('--auxk_coef', default=1/32, type=float, dest='auxk_coef',
                    help='auxk_coef')
parser.add_argument('--dead_threshold', default=30, type=int, dest='dead_threshold',
                    help='dead_threshold')
parser.add_argument('--hidden-size', default=None, type=int, dest='hidden_size',
                    help='the size of hidden layer')
parser.add_argument('--model_suffix', default=None, type=str, dest='model_suffix')


def get_current_k(global_step: int, initial_k: int, final_k: int, total_steps: int, k_decay_ratio: float = 0.7) -> int:
    """Get the current k value based on a linear decay schedule.
    
    Args:
        global_step: Current training step
        initial_k: Starting k value
        final_k: Target k value at the end of decay
        total_steps: Total number of training steps
        k_decay_ratio: Ratio of total steps when k reaches final_k (e.g., 0.7 means k reaches final_k at 70% of training)
    """
    k_decay_steps = int(total_steps * k_decay_ratio)
    
    if global_step >= k_decay_steps:
        return final_k
    
    progress = global_step / k_decay_steps
    return round(initial_k * (1 - progress) + final_k * progress)


def main():
    args = parser.parse_args()

    if not os.path.exists(args.pretrained_emb):
        print("pretrained_emb not found")
        stack_emb()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.use_ddp

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn(
                "nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.use_ddp:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.use_ddp:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> Creating CSR model, use pre-trained model '{}'".format(args.model_name))
    # Loading without the fc layer
    
    sota_backbone = timm.create_model(args.model_name, pretrained=True, num_classes=1000, )
    # state_dict = torch.load(args.backbone_ckpt, map_location='cpu')
    # sota_backbone.load_state_dict(state_dict)

    sota_backbone.eval()
    # 添加一个sae层，hidden_size = in_feature * 4
    if args.hidden_size is None:
        args.hidden_size = sota_backbone.fc.in_features * 4
    
    # 设置默认 initial_topk 值（如果启用 k-annealing 但未指定）
    if args.initial_topk is None:
        pass
    else:
        # 验证 initial_topk 的合理性
        max_safe_initial_topk = args.hidden_size // 4
        if args.initial_topk * 4 > args.hidden_size:
            print(f"Warning: initial_topk * 4 ({args.initial_topk * 4}) > hidden_size ({args.hidden_size})")
            args.initial_topk = max_safe_initial_topk
            print(f"Adjusted initial_topk to {args.initial_topk}")
        
        if args.initial_topk < args.topk:
            print(f"Warning: initial_topk ({args.initial_topk}) < final_topk ({args.topk})")
            args.initial_topk = args.topk
            print(f"Adjusted initial_topk to {args.initial_topk}")
    
    model = CSR(n_latents=args.hidden_size, topk=args.topk, auxk=args.auxk, dead_threshold=args.dead_threshold,
                normalize=False, n_inputs=sota_backbone.fc.in_features, pre_trained_backbone=sota_backbone)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using state single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                                  find_unused_parameters=True)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay,
                                 eps=6.25 * 1e-10)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=10, gamma=1)


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(5000, (3, 224, 224), 1000, transforms.ToTensor())  #
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, )

    else:
        train_dataset = CustomDataset(np.load(args.pretrained_emb))
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    # K-annealing 初始化
    if args.initial_topk is not None:
        # 计算总的训练步数
        total_steps = len(train_loader) * args.epochs
        # 设置初始 topk
        current_topk = args.initial_topk
        global_step = 0
        print(f"K-annealing enabled: initial_topk={args.initial_topk}, final_topk={args.topk}, total_steps={total_steps}, k_decay_ratio={args.k_decay_ratio}")
        # 更新模型的 topk 值
        if hasattr(model, 'module'):
            model.module.topk = current_topk
        else:
            model.topk = current_topk
    else:
        current_topk = args.topk
        global_step = 0
        total_steps = len(train_loader) * args.epochs

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        global_step = train(train_loader, model, criterion, optimizer, epoch, device, args, 
                          global_step, total_steps, current_topk)

        scheduler.step()

        # Save checkpoint for every epoch
        if args.rank == 0:
            file_name = f'{args.model_name}_CSR_topk_{args.topk}_auxk_{args.auxk}_auxk_coef_{args.auxk_coef}_cl_coef_{args.cl_coef}_lr_{args.lr}_hidden_{args.hidden_size}'
            if args.initial_topk is not None:
                file_name += f'_initial_topk_{args.initial_topk}_k_decay_{args.k_decay_ratio}'
            if args.use_label_CL:
                file_name += '_label_CL'
            if args.model_suffix:
                file_name += f'_{args.model_suffix}'
            ckpt_dir = 'ckpt'
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            checkpoint_path = os.path.join(ckpt_dir, f'{file_name}_epoch_{epoch}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, filename=checkpoint_path)
            print(f"Model saved to: {checkpoint_path}")

def normalized_mse(recon: torch.Tensor, xs: torch.Tensor, criterion) -> torch.Tensor:
    # only used for auxk
    xs_mu = xs.mean(dim=0)

    loss = criterion(recon, xs) / criterion(
        xs_mu[None, :].broadcast_to(xs.shape), xs
    )

    return loss

def inbatch_cl(latents_k_1: torch.Tensor, latents_k_2: torch.Tensor):
    """
    Original contrastive learning: only same sample's different representations are positive pairs
    """
    n_repeats = 2
    temperature = 0.2
    indexes = torch.arange(latents_k_1.shape[0])
    indexes = indexes.repeat(n_repeats).cuda()

    z = torch.concatenate((latents_k_1, latents_k_2), dim=0)
    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)

    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank():].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    loss_cl = -(torch.mean(torch.log(pos / (pos + neg))))
    return loss_cl


def label_based_cl(latents_k_1: torch.Tensor, latents_k_2: torch.Tensor, labels: torch.Tensor):
    """
    Label-based contrastive learning: samples with same labels are positive pairs
    
    Args:
        latents_k_1: First set of latent representations [batch_size, feature_dim]
        latents_k_2: Second set of latent representations [batch_size, feature_dim]  
        labels: Labels for each sample [batch_size]
    """
    temperature = 0.2
    batch_size = latents_k_1.shape[0]
    
    # Concatenate two sets of representations
    z = torch.concatenate((latents_k_1, latents_k_2), dim=0)  # [2*batch_size, feature_dim]
    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)  # Gather from all GPUs in distributed setting
    
    # Create label tensor for concatenated representations
    labels_concat = torch.concatenate((labels, labels), dim=0)  # [2*batch_size]
    gathered_labels = gather(labels_concat)  # Gather labels from all GPUs
    
    # Compute similarity matrix
    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)
    
    # Create positive and negative masks based on labels
    labels_concat = labels_concat.unsqueeze(1)  # [2*batch_size, 1]
    gathered_labels = gathered_labels.unsqueeze(0)  # [1, total_samples]
    
    # Positive mask: same labels (excluding self-comparison)
    pos_mask = (labels_concat == gathered_labels).float()
    # Remove diagonal elements to avoid self-comparison
    eye_mask = torch.eye(pos_mask.size(0), pos_mask.size(1), device=pos_mask.device)
    pos_mask = pos_mask - eye_mask
    
    # Negative mask: different labels
    neg_mask = (labels_concat != gathered_labels).float()
    
    # Compute positive and negative similarities
    pos = torch.sum(sim * pos_mask, dim=1)  # Sum over positive pairs
    neg = torch.sum(sim * neg_mask, dim=1)  # Sum over negative pairs
    
    # Avoid division by zero and log of zero
    pos = torch.clamp(pos, min=1e-8)
    neg = torch.clamp(neg, min=1e-8)
    
    # Compute contrastive loss
    loss_cl = -(torch.mean(torch.log(pos / (pos + neg))))
    
    return loss_cl

def train(train_loader, model, criterion, optimizer, epoch, device, args, global_step=0, total_steps=None, current_topk=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # K-annealing: 更新 topk 值
        if args.initial_topk is not None and total_steps is not None:
            new_topk = get_current_k(global_step, args.initial_topk, args.topk, total_steps, args.k_decay_ratio)
            if new_topk != current_topk:
                current_topk = new_topk
                # 更新模型的 topk 值
                if hasattr(model, 'module'):
                    model.module.topk = current_topk
                else:
                    model.topk = current_topk
                if global_step % 100 == 0:  # 每100步打印一次
                    print(f"Step {global_step}: Updated topk to {current_topk}")
        
        global_step += 1
        
        # measure data loading time
        data_time.update(time.time() - end)
        # move data to the same device as model
        images = images.to(torch.float32)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 使用当前的 topk 值
        effective_topk = current_topk if current_topk is not None else args.topk
        x_1, latents_pre_act_1, latents_k_1, recons_1, recons_aux_1 = model(images)
        x_3, latents_pre_act_3, latents_k_3, recons_3, recons_aux_3 = model(images, topk=4 * effective_topk)

        loss_k = criterion(x_1, recons_1)
        loss_4k = criterion(x_3, recons_3)
        loss_auxk = normalized_mse(recons_aux_1, x_1 - recons_1.detach() + model.pre_bias.detach(),
                                    criterion).nan_to_num(0)

        loss_cl = torch.tensor(0.0, device=device)
        if args.use_CL:
            x_2, latents_pre_act_2, latents_k_2, recons_2, recons_aux_2 = model(images)
            
            # Choose contrastive learning strategy
            if args.use_label_CL:
                # Check if labels are available and valid
                if target is not None and len(target.shape) > 0 and target.numel() > 0:
                    # Use label-based contrastive learning
                    loss_cl = label_based_cl(latents_k_1, latents_k_2, target)
                else:
                    # Fallback to original contrastive learning if no labels
                    print("Warning: use_label_CL is True but no valid labels found, using original inbatch_cl")
                    loss_cl = inbatch_cl(latents_k_1, latents_k_2)
            else:
                # Use original contrastive learning
                loss_cl = inbatch_cl(latents_k_1, latents_k_2)

        loss = loss_k + 0.125 * loss_4k + args.auxk_coef * loss_auxk + args.cl_coef * loss_cl

        losses.update(loss_k.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            if args.rank == 0:
                progress.display(i + 1)
    
    return global_step




def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
