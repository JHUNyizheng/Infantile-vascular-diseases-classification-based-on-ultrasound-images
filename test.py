from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import torch
import sys
import datetime
import torch.utils.data
import torch.nn as nn
import scipy.io as scio
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from opts import opts
from src.dataset.dataset_factory import get_dataset
from src.model.model import create_model, load_checkpoint, save_checkpoint
from src.util.lr_scheduler import get_scheduler
from src.util.lars import LARS
from src.util.logger import setup_logger
from src.util.util import AverageMeter
from src.util.data_parallel import DataParallel
from PIL import ImageFilter, ImageOps, Image
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.model.utils import PCA_svd, calculate_ssim, calculate_pnsr, calculate_rmse
from thop import profile
import cv2

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def maxmin_norm(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data


def plot_segmentation(number, opt, img, pred, label):
    color = [
        # blue 0
        [30, 144, 255],
        # green 1
        [0, 255, 0],
        # red 2
        [255, 0, 0],
        # Aqua 3
        [0, 255, 255],
        # pink 4
        [255, 0, 255],
        # yellow 5
        [255, 255, 0],
        # purple 6
        [128, 0, 255],
        # orange 7
        [255, 128, 0],
        # Blueviolet 8
        [138, 43, 226],
        # Grey 9
        [128, 128, 128],
        # Lavenderblush 10
        [255, 240, 245],
        # Lightgreen 11
        [144, 238, 144],
        # Lightsteelblue 12
        [176, 196, 222],
        # Mediumblue 13
        [0, 0, 205],
        # Crimson 14
        [220, 20, 60],
        # Darkolivegreen 15
        [85, 107, 47],
        # Deepskyblue 16
        [0, 191, 255],
        # Greenyellow 17
        [173, 255, 47],
        # Lavender 18
        [230, 230, 250],
        # Lightskyblue 19
        [135, 206, 250],
        # Mediumorchid 20
        [186, 85, 211],
        # Orchid 21
        [218, 112, 214],
        # Sienna 22
        [160, 82, 45],
        # Tomato 23
        [255, 99, 71]
    ]
    img = np.uint8(img * 255).transpose(1, 2, 0)

    # original_img = np.where(pred == 1, np.full_like(original_img, blue), original_img)
    # original_img = np.where(pred == 2, np.full_like(original_img, green), original_img)
    # original_img = np.where(pred == 3, np.full_like(original_img, red), original_img)
    # original_img = np.where(pred == 4, np.full_like(original_img, cyan), original_img)
    # original_img = np.where(pred == 5, np.full_like(original_img, pink), original_img)
    # original_img = np.where(pred == 6, np.full_like(original_img, yellow), original_img)
    # original_img = np.where(pred == 7, np.full_like(original_img, purple), original_img)
    pred = pred.flatten()
    label = label.flatten()
    pred[label == 0] = 0

    pred_result = np.zeros((pred.shape[0], 3))
    label_result = np.zeros((label.shape[0], 3))

    for i in range(0, 24):
        pred_result[np.where(pred == i), 0] = color[i][0]
        pred_result[np.where(pred == i), 1] = color[i][1]
        pred_result[np.where(pred == i), 2] = color[i][2]
        label_result[np.where(label == i), 0] = color[i][0]
        label_result[np.where(label == i), 1] = color[i][1]
        label_result[np.where(label == i), 2] = color[i][2]


    pred_result = np.uint8(np.reshape(pred_result, (512, 512, 3)))
    label_result = np.uint8(np.reshape(label_result, (512, 512, 3)))

    white = np.uint8(np.ones_like(img)[:, :10, :] * 255)

    result = np.concatenate([img, white, pred_result, white, label_result], axis=1)


    result = Image.fromarray(result).convert('RGB')
    plt.imshow(result)

    # save_dir = os.path.join("./img", opt.arch + '_' + opt.dataset)
    # # save_dir = os.path.join("./img", 'slotcon_' + opt.dataset)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # plt.savefig(os.path.join(save_dir, str(number) + '.png'))
    plt.show()

    print("OK")


def test_segmentation(model, train_loader, vaild_loader, logger, opt):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_vaild_meter = AverageMeter()
    acc_criterion = Accuracy("multiclass", num_classes=24, average="macro").cuda()
    end = time.time()
    train_len = len(train_loader)
    encoded_list = []
    label_list = []
    # for crops_a, coords, flags, crops_RGB in train_loader:
    for i, (img, acupoint, segm, seg_label, keypoint_label) in enumerate(train_loader):

        # crops = [crop.cuda(device=opt.gpus[0]) for crop in crops_a]
        # coords = [coord.cuda(device=opt.gpus[0]) for coord in coords]
        # flags = [flag.cuda(device=opt.gpus[0]) for flag in flags]
        # crops_RGB = [crops_R.cuda(device=opt.gpus[0]) for crops_R in crops_RGB]

        img = img.cuda(non_blocking=True).type(torch.float32)
        acupoint = acupoint.cuda(non_blocking=True).type(torch.float32)
        segm = segm.cuda(non_blocking=True).type(torch.float32)
        seg_label = seg_label.cuda(non_blocking=True).type(torch.float32)
        keypoint_label = keypoint_label.cuda(non_blocking=True).type(torch.float32)
        weights = torch.from_numpy(np.ones([15], np.float32)).cuda(non_blocking=True)


        masks, points = model(img, acupoint)

        # encoded_list.append(encoded.view(-1, encoded.shape[-1]).detach().cpu().numpy())
        # label_list.append(F.interpolate(label.unsqueeze(1).type(torch.float32), scale_factor=encoded.shape[1]/label.shape[1]).squeeze(1).type(torch.int64).view(-1).detach().cpu().numpy())
        pred = masks.argmax(dim=1)

        plot_segmentation(i, opt, img[0].detach().cpu().numpy(), pred[0].detach().cpu().numpy(), segm[0].detach().cpu().numpy())

def plot_color():
    color = [
        # blue 0
        [30, 144, 255],
        # green 1
        [0, 255, 0],
        # red 2
        [255, 0, 0],
        # Aqua 3
        [0, 255, 255],
        # pink 4
        [255, 0, 255],
        # yellow 5
        [255, 255, 0],
        # purple 6
        [128, 0, 255],
        # orange 7
        [255, 128, 0],
        # Blueviolet 8
        [138, 43, 226],
        # Grey 9
        [128, 128, 128],
        # Lavenderblush 10
        [255, 240, 245],
        # Lightgreen 11
        [144, 238, 144],
        # Lightsteelblue 12
        [176, 196, 222],
        # Mediumblue 13
        [0, 0, 205],
        # Crimson 14
        [220, 20, 60],
        # Darkolivegreen 15
        [85, 107, 47],
        # Deepskyblue 16
        [0, 191, 255],
        # Greenyellow 17
        [173, 255, 47],
        # Lavender 18
        [230, 230, 250],
        # Lightskyblue 19
        [135, 206, 250],
        # Mediumorchid 20
        [186, 85, 211],
        # Orchid 21
        [218, 112, 214],
        # Sienna 22
        [160, 82, 45],
        # Tomato 23
        [255, 99, 71]
    ]

    color_all = np.zeros([240, 240, 3])

    for i in range(0, 24):
        color_all[:, i*10:i*10+10, 0] = color[i][0]
        color_all[:, i*10:i*10+10, 1] = color[i][1]
        color_all[:, i*10:i*10+10, 2] = color[i][2]

    color_all = np.uint8(color_all)

    color_all = Image.fromarray(color_all).convert('RGB')
    plt.imshow(color_all)

    plt.savefig(os.path.join('color.png'))
    plt.show()

    # print("OK")


def main(opt):
    """ Dataset and loader """
    opt.local_rank = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda:1')
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=1, rank=0)

    # if os.environ["LOCAL_RANK"] is not None:
    #     opt.local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.device_count()
    # torch.cuda.set_device(opt.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # cudnn.benchmark = True

    opt.world_size = dist.get_world_size()
    opt.batch_size = int(opt.batch_size / opt.world_size)

    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="HSNet")

    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(opt)).items()))
    )

    Dataset = get_dataset(opt.dataset)
    train_dataset = Dataset('train', opt=opt)
    valid_dataset = Dataset('val', opt=opt)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    vaild_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    print("loading trainset...")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=opt.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )
    vaild_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=opt.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=opt.num_workers,
        sampler=vaild_sampler,
        pin_memory=True,
        drop_last=True
    )

    opt.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {opt.num_instances}")

    # create model
    logger.info("=> creating model '{}'".format(opt.arch))
    # if opt.finetune == 'super_Resolution':
    #     opt.image_size = 64

    model = create_model(opt=opt).cuda()
    # macs, params = profile(model, inputs=(input1,))
    #
    # print("FLOPS:", str(2 * macs))
    # print("params:", str(params))

    model = model.cuda()


    if opt.load_model != '':
        load_checkpoint(logger, opt, model, type='plot')


    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], find_unused_parameters=True)


    # logger.info(model)
    test_segmentation(model, train_loader, vaild_loader, logger, opt)

    # for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    #     train_sampler.set_epoch(epoch)
    #     # start_time = time.time()
    #     train(model, train_loader, vaild_loader, optimizer, scaler, scheduler, epoch, logger, opt)
    #
    #     if dist.get_rank() == 0 and (epoch % int(opt.save_freq) == 0 or epoch in opt.save_point):
    #         save_checkpoint(logger, opt, epoch, model, optimizer, scheduler, scaler)

        # end_time = time.time()
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        #
        # data_str = f'{opt.exp_id} | {opt.arch} | Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        # data_str += f'\t{opt.exp_id} | {opt.arch} | Best Valid Loss: {best_valid_loss:.4f}\n'
        # # data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        # print(data_str)


if __name__ == '__main__':
    opt = opts().parse()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    # opt.exp_id = "test"
    # plot_color()
    """ Create a directory. """
    if opt.exp_id == 'default':
        print("exp_id null !!!")
        sys.exit(1)
    else:
        opt.output_dir = os.path.join('..', "results", opt.arch, opt.exp_id)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    main(opt)

