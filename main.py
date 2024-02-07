# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import copy
from HED import HED_paired_dataset
import numpy as np
from typing import Any, Callable, Optional, Tuple

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')#0
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.3, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--projector', default='512-512-512', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--predictor', default='512', type=str,
                    metavar='MLP', help='predictor MLP')                    
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--warmup', default=10, type=int)
parser.add_argument('--local-crops-number', default=0, type=int)
parser.add_argument('--asym-loss', action='store_true')
parser.add_argument('--aug-s', default=1., type=float)
parser.add_argument('--aug-asym-off', action='store_true')
parser.add_argument('--original-multicrop', action='store_true')
parser.add_argument('--checkpoint-dir', default='./checkpoint_proposed/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--loss', default='cos', type=str)
parser.add_argument('--pred-lr-rate', default=10, type=float)
parser.add_argument('--weight-decay-filter-off', action='store_false')
parser.add_argument('--optimizer',default='lars', type=str)

def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://127.0.0.1:%d'%random.randint(10000, 60000)
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    #data augmentation and split

#######################################################################
### Section modified by Achille Rieu - start
#######################################################################
    transform = Transform(args, local_crops_number=args.local_crops_number)
    dataset = dataset_HED(args.data, split='train+unlabeled', download=False,
        transform=transform
    )

#######################################################################
### Section modified by Achille Rieu - end
#######################################################################
  
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = MultiEpochsDataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    model = SimSiam(args).cuda(gpu)
    if args.rank==0:
        print(model)
        print(model, file=stats_file)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_enc = []
    param_pred = []
    for name, param in model.named_parameters():
        if 'predictor' in name:
            param_pred.append(param)
        else:
            param_enc.append(param)
    parameters = [{'params': param_enc}, {'params': param_pred}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    if args.optimizer == 'lars':
        optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=args.weight_decay_filter_off,
                        lars_adaptation_filter=True)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=0, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=0, weight_decay=args.weight_decay, eps=1e-8)
    else:
        raise NotImplementedError()

    if args.rank==0:
        print(optimizer)
        print(optimizer, file=stats_file)

    scaler = torch.cuda.amp.GradScaler()    
    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
    else:
        start_epoch = 0

    start_time = time.time()
    acc1, acc5 = torch.zeros(1), torch.zeros(1)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

#######################################################################
### Section modified by Achille Rieu - start
#######################################################################
      
        for step, (y1, y2, labels) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                ssl_loss, cls_loss, z, cls_pred = model(y1, y2, labels)

#######################################################################
### Section modified by Achille Rieu - end
#######################################################################
          
            labels = labels[labels!=-1]
            scaler.scale(ssl_loss+cls_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if step % args.print_freq == 0:
                if args.rank == 0:
                    if len(labels)>0:
                        acc1, acc5 = accuracy(
                                output=cls_pred,
                                target=labels,
                                topk=(1,5))
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_pred=optimizer.param_groups[1]['lr'],
                                 ssl_loss=ssl_loss.item(),
                                 cls_loss=cls_loss.item(),
                                 acc1=acc1.item(),
                                 acc5=acc5.item(),
                                 avg_length=z.norm(dim=1).mean().item(),
                                 std_unit_z1=nn.functional.normalize(z,dim=1).std(dim=0).mean().item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         scaler=scaler.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
            if epoch%10==0:
                torch.save(state, str(args.checkpoint_dir) +'/checkpoint_ep%d.pth'%epoch)
        if torch.isnan(ssl_loss+cls_loss+nn.functional.normalize(z,dim=1).std(dim=0).mean()):
            raise RuntimeError('NaN')
    if args.rank == 0:
        # save final model
        torch.save(state, str(args.checkpoint_dir) +'/checkpoint_ep%d.pth'%epoch)
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet18.pth')
        torch.save(model.module.backbone2.state_dict(),
                   args.checkpoint_dir / 'resnet18_backbone_hed.pth')
    

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = args.warmup * len(loader)
    base_lr = args.batch_size / 256

    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
        optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    else:
        q = 0.5*(1+ math.cos(math.pi * (step-warmup_steps) / (max_steps-warmup_steps)))
        end_lr = base_lr * 0.00
        lr = base_lr * q + end_lr * (1 - q)
        optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = base_lr * args.learning_rate_weights*args.pred_lr_rate
        


def accuracy(output, target, topk=(1, )):
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
def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass
#######################################################################
### Section modified by Achille Rieu - start
#######################################################################
class dataset_HED(torchvision.datasets.STL10):
    base_folder_hed = "stl10_hed_npy/"

    def __init__(self, root: str, split: str = "train", folds: Optional[int] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, folds, transform, target_transform, download)

        path_to_data_hed = os.path.join(self.root, self.base_folder_hed)
        self.data_hed = HED_paired_dataset(self.data, path_to_data_hed, split)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        target: Optional[int]
        if self.labels is not None:
            img, img_hed, target = self.data[index], self.data_hed[index], int(self.labels[index])
        else:
            img, target = self.data[index], self.data_hed[index], None

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img_hed = Image.fromarray(img_hed)

        if self.transform is not None:
            img, img_hed = self.transform(img, img_hed)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, img_hed, target)

class SimSiam(nn.Module):
    # def __init__(self, model):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.backbone2 = copy.deepcopy(self.backbone)
        self.backbone2.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      
        # projector
        sizes = [512] + list(map(int, args.projector.split('-')))
        layers = []
        if len(sizes)==1:
            layers.append(nn.Linear(sizes[-1], sizes[-1], bias=False))        
        else:
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))        
        self.projector = nn.Sequential(*layers)
        self.projector2 = copy.deepcopy(self.projector)

#######################################################################
### Section modified by Achille Rieu - end
#######################################################################
        
        # predictor
        sizes = [sizes[-1]] + list(map(int, args.predictor.split('-')))
        layers = []
        if len(sizes)==1:
            layers.append(nn.Linear(sizes[-1], sizes[-1], bias=False))        
        else:
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))        
        self.predictor = nn.Sequential(*layers)

        self.classifier = nn.Linear(512, 10)        
        cossim = nn.CosineSimilarity(dim=1)
        # loss = RegCos()
        self.loss = lambda x,y: -cossim(x,y).mean()
        self.ce = nn.CrossEntropyLoss()

#######################################################################
### Section modified by Achille Rieu - start
#######################################################################
  
    def compute_losses(self, ps, zs):
        # SSL Losses
        ssl_loss = torch.add(torch.mul(self.loss(ps[0], zs[1].detach()), 0.5), torch.mul(self.loss(ps[1], zs[0].detach()), 0.5))
        return ssl_loss

        
    def forward(self, y1, y2, labels):
        # Zs
        t = self.backbone(y1)
        cls_pred = self.classifier(t.detach()) # Note: no gradient flow from classifier
        z1 = self.projector(t)
        z2 = self.projector2(self.backbone2(y2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        zs = [z1,z2]
        ps = [p1,p2]
                
        ssl_loss = self.compute_losses(ps, zs) 

        # CLS loss 
        # # STL10 is a dataset for semi-suprvised learning, some of the training data doesn't have labels.
        # # The code below  filters out those unlabeled data.
        cls_pred = cls_pred[labels!=-1, :]
        labels = labels[labels!=-1]
        if len(cls_pred)>0:
            cls_loss = self.ce(cls_pred, labels)
        else:
            cls_loss = 0*self.classifier.weight.mean() + 0*self.classifier.bias.mean()

        return ssl_loss, cls_loss, zs[0], cls_pred

#######################################################################
### Section modified by Achille Rieu - end
#######################################################################

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])
                
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

#######################################################################
### Section modified by Achille Rieu - start
#######################################################################

class Transform:
    def __init__(self, args, global_crop_scale=(0.08,1.0), local_crops_scale=(0.05, 0.4), local_crops_number=0):
        
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4*args.aug_s, contrast=0.4*args.aug_s,
                                        saturation=0.2*args.aug_s, hue=0.1*args.aug_s)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])

        normalize_hed = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=0.449,
                                std=0.226)
            ])

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crop_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            Solarization(p=0.0),            
            normalize
        ])

        self.global_transform_hed = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=global_crop_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            Solarization(p=0.0),            
            normalize_hed
        ])

        self.local_crops_number = local_crops_number
        


    def __call__(self, x, x_hed):
        return self.global_transform(x), self.global_transform_hed(x_hed)
      
#######################################################################
### Section modified by Achille Rieu - end
#######################################################################

if __name__ == '__main__':
    main()
