from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import shutil
import torchvision.models as models
import torch
import torch.nn as nn
import os

from .IHVMNet import IHVMNet
_network_factory = {
    'ihvmnet': IHVMNet,
}


def create_model(opt=None):
    # QuadT = QuadTree(min_patch_size=16, max_patch_size=64, num_patches=100, num_scales=3, opt=opt),
    Model_List = _network_factory[opt.arch](opt)

    return Model_List


def load_checkpoint(logger, opt, model, optimizer=None, scheduler=None, scaler=None, type='train'):
    if os.path.isfile(opt.load_model):
        logger.info(f"=> loading checkpoint '{opt.load_model}'")
        checkpoint = torch.load(opt.load_model, map_location='cpu')

        opt.start_epoch = checkpoint['epoch'] + 1
        if type == 'finetune':
            new_model = {}
            for k, v in checkpoint['model'].items():
                # if 'lms' not in k:
                if 'build_neck' not in k and 'build_box_head' not in k:
                    # k = k.replace("module.", "")
                    new_model[k] = v
                # else:
                #     k = k.replace("module.", "")
                #     new_model[k] = v

            model.load_state_dict(new_model, False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            # #
            # if opt.fp16 and 'scaler' in checkpoint:
            #     scaler.load_state_dict(checkpoint['scaler'])

        elif type == 'train':
            # model.load_state_dict(checkpoint['model'], False)
            new_model = {}
            for k, v in checkpoint['model'].items():
                new_model[k] = v

            model.load_state_dict(new_model)
            # model.load_state_dict(checkpoint['model'], False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            #
            if opt.fp16 and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            # new_model = {}
            # for k, v in checkpoint['model'].items():
            #     if 'to_patch_embedding' in k:
            #         k = k.replace("to_patch_embedding", "")
            #         new_model[k] = v
            # model.load_state_dict(new_model, False)
        elif type == 'plot':
            # model.module.re_init(opt)
            #
            new_model = {}
            for k, v in checkpoint['model'].items():
                k = k.replace("module.", "")
                new_model[k] = v

            model.load_state_dict(new_model)
            # model.load_state_dict(checkpoint['model'])

        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        #
        # if opt.fp16 and 'scaler' in checkpoint:
        #     scaler.load_state_dict(checkpoint['scaler'])

        logger.info("=> loaded checkpoint '{}' (epoch {})".format(opt.load_model, checkpoint['epoch']))

    else:
        logger.info("=> no checkpoint found at '{}'".format(opt.load_model))

# def save_model(path, epoch, model, optimizer=None):
#     if isinstance(model, torch.nn.DataParallel):
#         state_dict = model.module.state_dict()
#     else:
#         state_dict = model.state_dict()
#     data = {'epoch': epoch,
#             'state_dict': state_dict}
#     if not (optimizer is None):
#         data['optimizer'] = optimizer.state_dict()
#     torch.save(data, path)

def save_checkpoint(logger, opt, epoch, model, optimizer, scheduler, scaler=None):
    logger.info('==> Saving...')
    state = {
        'args': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if opt.fp16:
        state['scaler'] = scaler.state_dict()

    file_name = os.path.join(opt.output_dir, 'current.pth')
    torch.save(state, file_name)
    # file_name = os.path.join(opt.output_dir, f'ckpt_epoch_{epoch}.pth')
    # torch.save(state, file_name)
    # shutil.copyfile(file_name, os.path.join(opt.output_dir, 'current.pth'))