from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json
import torch

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser("HSNet")
        # basic experiment setting
        self.parser.add_argument('--dataset', default='luojiassr', help='luojiassr | houston2018')
        self.parser.add_argument('--test_dataset', default='', help='real | simulation')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--no_pause', action='store_true')
        self.parser.add_argument('--load_model', default='', help='path to pretrained model')
        # self.parser.add_argument('--step', default='1', help='Training step')

        # system
        self.parser.add_argument('--gpus', default='2,3', help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num-workers', type=int, default=0, help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true', help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317, help='random seed')  # from CornerNet
        self.parser.add_argument('--not_set_cuda_env', action='store_true', help='used when training in slurm clusters.')

        # model
        self.parser.add_argument('--arch', default='ihvmnet', help='model name')
        self.parser.add_argument('--patch_size', default=16, help='patch size')
        self.parser.add_argument('--in_chans', default=3, help='RGB channel')
        self.parser.add_argument('--embed_dim', default=768, help='ViT embed_dim')
        self.parser.add_argument('--depth', default=12, help='ViT depth')
        self.parser.add_argument('--num_heads', default=12, help='ViT head number')
        self.parser.add_argument('--mlp_ratio', default=4.0, help='mlp_ratio')
        self.parser.add_argument('--out_chans', default=256, help='out chans')

        self.parser.add_argument('--prompt_embed_dim', default=256, help='prompt_embed_dim = out_chans')

        # QuadTree
        self.parser.add_argument('--min_patch_size', type=int, default=16, help='QuadTree min patch size')
        self.parser.add_argument('--max_patch_size', type=int, default=64, help='QuadTree max patch size')
        self.parser.add_argument('--num_patches', type=int, default=100, help='patch number')
        self.parser.add_argument('--num_scales', type=int, default=3, help='input image channels')

        # input
        self.parser.add_argument('--image_size', type=int, default=512, help='image crop size')
        self.parser.add_argument('--num_classes', type=int, default=15, help='image crop size')

        # train
        self.parser.add_argument('--optimizer', default='adam', choices=['sgd', 'lars', 'adam'], help='optimizer choice')
        self.parser.add_argument('--base-lr', type=float, default=1.0, help='base learning when batch size = 256. final lr is determined by linear scale')
        self.parser.add_argument('--num_epochs', type=int, default=400, help='total training epochs.')
        self.parser.add_argument('--start_epoch', type=int, default=1, help='used for resume')
        self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        self.parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
        self.parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
        self.parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='whether or not to turn on automatic mixed precision')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,help='batch size on the master gpu.')
        self.parser.add_argument('--save_point', type=str, default='10,120', help='when to save the model to disk.')
        self.parser.add_argument('--save-freq', type=str, default='30', help='when to save the model to disk.')
        self.parser.add_argument('--auto-resume', action='store_true', help='auto resume from current.pth')
        self.parser.add_argument('--print-freq', type=int, default=10, help='print frequency')



        # test
        self.parser.add_argument('--flip-test', action='store_true', help='flip data augmentation.')
        self.parser.add_argument('--save-log', default='False', help='Save or not')

        #mask
        self.parser.add_argument('--mim_masking_ratio', type=float, default=0.7, help='print frequency')
        self.parser.add_argument('--mim_intermediate_losses', type=bool, default=False, help='Number of fusion feature layers')
        self.parser.add_argument('--mim_mask_patch_size', type=int, default=4, help='finetune type')
        self.parser.add_argument('--spectral_pos_embed', type=bool, default=False, help='e')
        self.parser.add_argument('--blockwise_patch_embed', type=bool, default=True, help='e')
        self.parser.add_argument('--spectral_only', type=bool, default=False, help='finetune type')
        self.parser.add_argument('--to_pixels_per_spectral_block', type=bool, default=False, help='finetune type')
        self.parser.add_argument('--tube_masking', type=bool, default=True, help='finetune type')


    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        if opt.test_dataset == '':
            opt.test_dataset = opt.dataset

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        # opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
        # opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.save_point = [int(i) for i in opt.save_point.split(',')]
        # opt.num_workers = max(opt.num_workers, 2 * len(opt.gpus))



        # # log dirs
        # opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        # opt.data_dir = os.path.join(opt.root_dir, 'data')
        # opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        # opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        # opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        #
        # if opt.resume and opt.load_model == '':
        #     opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
        return opt


    def init(self, args=''):
        print("OK")

