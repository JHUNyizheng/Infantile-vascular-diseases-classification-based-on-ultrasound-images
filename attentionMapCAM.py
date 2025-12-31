from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json, cv2
import torch
import sys
import datetime
import torch.utils.data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
from opts import opts
from src.dataset.dataset_factory import get_dataset, my_collate
from src.model.model import create_model, load_checkpoint, save_checkpoint
from src.util.lr_scheduler import get_scheduler
from src.util.lars import LARS
from src.util.logger import setup_logger
from src.util.pytorch_grad_cam import GradCAM
from src.util.pytorch_grad_cam.utils.image import show_cam_on_image
from src.util.util import AverageMeter
# from chainercv.evaluations import eval_semantic_segmentation
from src.model.loss import Focal_Loss, CE_Loss, v8DetectionLoss
from src.model.utils import calculate_mse, calculate_rmse, calculate_nmse
from src.tools.val import DetectionValidator
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
from PIL import Image
""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_optimizer(opt, model):

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt.batch_size * opt.world_size / 256 * opt.base_lr,
            # lr=opt.base_lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':

        optimizer = torch.optim.AdamW(
            model.parameters(),
            # lr=opt.batch_size * opt.world_size / 256 * opt.base_lr)
            lr=0.0001)
    elif opt.optimizer == 'lars':
        optimizer = LARS(
            model.parameters(),
            lr=opt.batch_size * opt.world_size / 256 * opt.base_lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay)


    return optimizer


def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape  # 1,960,7,7
    output_cam = []
    for idx in class_idx:  # 只输出预测概率最大值结果不需要for循环
        feature_conv = feature_conv.reshape((nc, h * w))  # [960,7*7]
        cam = weight_softmax[idx].dot(
            feature_conv.reshape((nc, h * w)))  # (5, 960) * (960, 7*7) -> (5, 7*7) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
        cam_img = np.uint8(255 * cam_img)  # Format as CV_8UC1 (as applyColorMap required)

        # output_cam.append(cv2.resize(cam_img, size_upsample))  # Resize as image size

        output_cam.append(cam_img)
    return output_cam


def get_pytorch_model_info(model: torch.nn.Module) -> (dict, list):
    """
    输入一个PyTorch Model对象，返回模型的总参数量（格式化为易读格式）以及每一层的名称、尺寸、精度、参数量、是否可训练和层的类别。

    :param model: PyTorch Model
    :return: (总参数量信息, 参数列表[包括每层的名称、尺寸、数据类型、参数量、是否可训练和层的类别])
    """
    params_list = []
    total_params = 0
    total_params_non_trainable = 0


def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# 自动找到最后一个卷积层并注册钩子
def find_last_conv_layer_in_pipeline(module):
    last_conv = None
    for name, layer in module.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv = layer
    return last_conv


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        # return (model_output[self.category, :, :] * self.mask).sum()
        return (model_output[self.category] * self.mask).sum()

def reshape_transform(input):
    return input.view(-1, 20, 20, 768).permute(0, 3, 1, 2).contiguous()

def main(opt):
    """ Dataset and loader """
    opt.local_rank = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23455', world_size=1, rank=0)

    # if os.environ["LOCAL_RANK"] is not None:
    #     opt.local_rank = int(os.environ["LOCAL_RANK"])
    # # os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
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
        drop_last=True,
        collate_fn=my_collate
    )

    vaild_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=opt.batch_size,
        # batch_size=int(opt.batch_size / 2),
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=opt.num_workers,
        sampler=vaild_sampler,
        pin_memory=True,
        drop_last=True,
        collate_fn=my_collate
    )

    # opt.num_instances = len(train_loader.dataset)
    opt.num_instances_val = len(vaild_loader.dataset)
    # logger.info(f"length of training dataset: {opt.num_instances}")
    logger.info(f"length of val dataset: {opt.num_instances_val}")

    # create model
    logger.info("=> creating model '{}'".format(opt.arch))
    model = create_model(opt=opt).cuda()
    # input1 = torch.randn(1, 250, 256, 256)
    #
    # macs, params = profile(model, inputs=(input1,))
    #
    # print("FLOPS:", str(2 * macs))
    # print("params:", str(params))

    # if opt.step == '1':
    #     for name, param in model.named_parameters():
    #         if "build_box_head" in name or 'build_neck' in name:
    #             param.requires_grad = False
    # elif opt.step == '2':
    #     for name, param in model.named_parameters():
    #         if "build_box_head" in name or 'build_neck' in name:
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if "build_box_head" in name or 'build_neck' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    # model = torch.nn.DataParallel(model, device_ids=opt.gpus)  # 指定要用到的设备
    # model = DataParallel(model, device_ids=opt.gpus, chunk_sizes=opt.chunk_sizes).cuda(device=opt.gpus[0])  # 指定要用到的设备
    # 自动找到最后一个卷积层并注册钩子
    last_conv_layer = find_last_conv_layer_in_pipeline(model.build_pipeline)
    if last_conv_layer is not None:
        last_conv_layer.register_forward_hook(forward_hook)
        last_conv_layer.register_backward_hook(backward_hook)
    else:
        raise ValueError("No convolutional layer found in build_pipeline.")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], find_unused_parameters=True)


    # logger.info(model)

    # create optimizer
    optimizer = get_optimizer(opt, model)

    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.to(device=opt.device, non_blocking=True)

    # define scheduler
    scheduler = get_scheduler(optimizer, len(train_loader), opt)

    # define scaler
    if opt.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if opt.load_model != '':
        # opt.load_model = os.path.join(opt.load_model, "ckpt_epoch_10.pth")
        # load_checkpoint(logger, opt, model, optimizer, scheduler, scaler, type='finetune')
        load_checkpoint(logger, opt, model, optimizer, scheduler, scaler)

    print("start...")

    # model_features = nn.Sequential(*list(model.children())[:-2])
    # fc_weights = model.state_dict()  # numpy数组取维度fc_weights[0].shape->(5,960)
    class_num = {0: 'NR', 1: 'VM', 2: 'IH'}
    model.eval()
    # model_features.eval()
    step = 0

    for i, (image_gray, image_color, gray_coords, phlebolith_labels, classes) in enumerate(train_loader):
        # if i > 5:
        #     break
        # image_color = torch.zeros_like(image_color)
        step = step + 1
        img = image_gray[0][0]
        img_c = image_color[0]
        image_gray = image_gray.cuda(non_blocking=True).type(torch.float32)
        image_color = image_color.cuda(non_blocking=True).type(torch.float32)
        input = torch.cat([image_gray, image_color], dim=1)
        # classes = classes.cuda(non_blocking=True).type(torch.float32)
        # compute output and loss

        # features = model_features(image_gray).detach().cpu().numpy()


        # # 确保输入张量需要梯度
        # image_gray.requires_grad = True
        # image_color.requires_grad = True

        box_pred, class_pred = model(input)

        for cls in range(3):
            round_category = cls
            round_mask = torch.argmax(class_pred[0], dim=0).detach().cpu().numpy()
            round_mask_float = np.array(np.float32(round_mask == round_category))

            # target_layers = [model.module.build_MSEs_AMM.MSEs[0].attn.attn[-1]]
            target_layers = [model.module.build_pipeline.norm3]
            # target_layers = [model.module.image_encoder.neck[-1]]
            targets = [SemanticSegmentationTarget(round_category, round_mask_float)]
            grad_cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
            grayscale_cam = grad_cam(input_tensor=input, targets=targets)[0, :]

            rgb_img = input[0, :input.shape[1] // 2, :, :].detach().cpu().numpy()
            cam_image = show_cam_on_image(rgb_img.transpose(1, 2, 0), grayscale_cam, use_rgb=True)
            img = Image.fromarray(cam_image)
            plt.title(class_num[int(round_mask.item())])
            plt.imshow(img)

            save_dir = os.path.join("./attentionResult_train", opt.dataset + '_' + class_num[round_category])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, str(step) + '.png'))
            plt.clf()
            # plt.show()

        # # 获取预测的类别
        # pred_class = class_pred.argmax(dim=1).item()
        #
        # # 反向传播以获取梯度
        # class_pred[:, pred_class].backward()
        #
        # # 检查gradients是否为None，如果为None则说明backward_hook未正确触发
        # if gradients is None:
        #     raise ValueError(
        #         "Gradient is None. Make sure that backward_hook is properly registered and backward pass is correctly executed.")
        #
        # # 计算权重并生成CAM
        # weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        # cam = torch.sum(weights * features, dim=1).squeeze().detach().cpu().numpy()
        # # cam = np.maximum(cam, 0)  # 取ReLU
        # cam = cv2.resize(cam, (640, 640))
        #
        #
        # # cam1 = score[0][0][0].detach().cpu().numpy()
        # # cam2 = score[1][0][0].detach().cpu().numpy()
        #
        # cam = (cam - cam.min()) / (cam.max() - cam.min())
        # # cam1 = (cam1 - cam1.min()) / (cam1.max() - cam1.min())
        # # cam2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min())
        #
        # # cam1 = (1 + cam1) * cam
        # # cam2 = (1 + cam2) * cam
        #
        # # cam = np.where(cam1>cam2, 1,0)
        # # cam_b = 1-cam
        #
        # # 归一化并可视化热力图
        # cam = (cam - cam.min()) / (cam.max() - cam.min())
        # # cam1 = (cam1 - cam1.min()) / (cam1.max() - cam1.min())
        # # cam2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min())
        # heatmap1 = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # img_rgb = image_gray.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # img_rgb = (img_rgb * 255).astype(np.uint8)
        # if img_rgb.shape[:2] != heatmap1.shape[:2]:
        #     heatmap1 = cv2.resize(heatmap1, (img_rgb.shape[1], img_rgb.shape[0]))
        # superimposed_img_gray = cv2.addWeighted(img_rgb, 0.6, heatmap1, 0.4, 0)
        #
        # heatmap2 = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # img_rgb = image_color.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # img_rgb = (img_rgb * 255).astype(np.uint8)
        # if img_rgb.shape[:2] != heatmap2.shape[:2]:
        #     heatmap2 = cv2.resize(heatmap2, (img_rgb.shape[1], img_rgb.shape[0]))
        # superimposed_img_color = cv2.addWeighted(img_rgb, 0.6, heatmap2, 0.4, 0)
        #
        # # # 显示原始图像和热力图
        # # plt.figure(figsize=(10,16))
        # # plt.subplot(3, 2, 1)
        # # plt.imshow(img, cmap='gray')
        # # # plt.colorbar()
        # # plt.title('Gray Image')
        # # plt.axis('off')
        # #
        # # plt.subplot(3, 2, 2)
        # # plt.imshow(np.transpose(img_c,(1,2,0)))
        # # # plt.colorbar()
        # # plt.title('Color Image')
        # # plt.axis('off')
        # #
        # # plt.subplot(3, 2, 3)
        # # # plt.imshow(superimposed_img_gray[..., ::-1])  # 转换回BGR用于显示
        # # plt.imshow(cam1, cmap='hot')  # 转换回BGR用于显示
        # # # plt.colorbar()
        # # plt.title('Score')
        # # plt.axis('off')
        # #
        # #
        # # plt.subplot(3, 2, 4)
        # # # plt.imshow(superimposed_img_color[..., ::-1])  # 转换回BGR用于显示
        # # plt.imshow(cam2, cmap='hot')  # 转换回BGR用于显示
        # # plt.colorbar()
        # # plt.title('Score')
        # # plt.axis('off')
        # #
        # # plt.subplot(3, 2, 5)
        # # # plt.imshow(superimposed_img_gray[..., ::-1])  # 转换回BGR用于显示
        # # plt.imshow(cam, cmap='hot')  # 转换回BGR用于显示
        # # # plt.colorbar()
        # # plt.title('Class(Predict/Label):'+ class_num[pred_class]+'/'+ class_num[
        # #     int(classes[0][0].detach().cpu().numpy())])
        # # plt.axis('off')
        # #
        # #
        # #
        # # plt.subplot(3, 2, 6)
        # # # plt.imshow(superimposed_img_color[..., ::-1])  # 转换回BGR用于显示
        # # plt.imshow(cam_b, cmap='hot')  # 转换回BGR用于显示
        # # plt.colorbar()
        # # plt.title('Class(Predict/Label):' + class_num[pred_class] + '/' + class_num[
        # #     int(classes[0][0].detach().cpu().numpy())])
        # # plt.axis('off')
        #
        # # 显示原始图像和热力图
        # plt.figure(figsize=(10, 10))
        # plt.subplot(2, 2, 1)
        # plt.imshow(img, cmap='gray')
        # # plt.colorbar()
        # plt.title('Gray Image')
        # plt.axis('off')
        #
        #
        #
        # plt.subplot(2, 2, 2)
        # # plt.imshow(superimposed_img_gray[..., ::-1])  # 转换回BGR用于显示
        # plt.imshow(superimposed_img_gray, cmap='hot')  # 转换回BGR用于显示
        # # plt.colorbar()
        # plt.title('Class(Predict/Label):' + class_num[pred_class] + '/' + class_num[
        #     int(classes[0][0].detach().cpu().numpy())])
        # plt.axis('off')
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(np.transpose(img_c, (1, 2, 0)))
        # # plt.colorbar()
        # plt.title('Color Image')
        # plt.axis('off')
        #
        # plt.subplot(2, 2, 4)
        # # plt.imshow(superimposed_img_color[..., ::-1])  # 转换回BGR用于显示
        # plt.imshow(superimposed_img_color, cmap='hot')  # 转换回BGR用于显示
        # # plt.colorbar()
        # plt.title('Class(Predict/Label):' + class_num[pred_class] + '/' + class_num[
        #     int(classes[0][0].detach().cpu().numpy())])
        # plt.axis('off')
        #
        #
        # # plt.savefig('/home/dell/Project/IHNet/runs/plot_val_CAM_4/'+str(i)+'-'+ class_num[pred_class]+'-'+ class_num[int(classes[0][0].detach().cpu().numpy())]+'.png')
        # plt.savefig('/home/dell/Project/IHNet/runs/plot_val_CAM_4_noColor/'+str(i)+'-'+ class_num[pred_class]+'-'+ class_num[int(classes[0][0].detach().cpu().numpy())]+'.png')
        #
        # # plt.show()
        # plt.close()  # 清除当前图形
        #
        # print(i)

        # for name, child_modle in model.named_children():
        #     print(name, ":")
        #     # print(name, child_modle, ":")
        #     for name, param in child_modle.named_parameters():
        #         print("    ", name, param.shape)



    # for i, (image_gray, image_color, gray_coords, phlebolith_labels, classes) in enumerate(vaild_loader):
    #     if i > 1:
    #         break
    #     image_gray_1 = np.transpose(image_gray[0], (1, 2, 0))
    #     plt.imshow(image_gray_1)
    #     plt.show()
    #     # plt.imshow(image_color[0])
    #     # plt.show()
    #
    #     features = model_features(image_gray).detach().cpu().numpy()
    #     result = features * 0.3 + image_gray.detach().cpu().numpy() * 0.7
    #     hmap = np.transpose(result[0], (1, 2, 0))
    #
    #     plt.imshow(hmap)
    #     plt.show()




    # start_epoch = 0
    # """ Training the model """
    # for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    #     opt.now_epcho = epoch
    #     train_sampler.set_epoch(epoch)


        # start_time = time.time()
        # train(model, train_loader, vaild_loader, optimizer, scaler, scheduler, epoch, logger, opt)

        # save_checkpoint(logger, opt, epoch, model, optimizer, scheduler, scaler)

        # if dist.get_rank() == 0 and (epoch % int(opt.save_freq) == 0 or epoch in opt.save_point):
        #     save_checkpoint(logger, opt, epoch, model, optimizer, scheduler, scaler)
        #
        # if dist.get_rank() == 0 and epoch % int(opt.save_freq) == 0:
        #     logger.info('==> Saving...')
        #     state = {
        #         'args': opt,
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #         'epoch': epoch,
        #     }
        #     if opt.fp16:
        #         state['scaler'] = scaler.state_dict()
        #     file_name = os.path.join(opt.output_dir, f'ckpt_epoch_{epoch}.pth')
            # torch.save(state, file_name)

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

    """ Create a directory. """
    if opt.exp_id == 'default':
        print("exp_id null !!!")
        sys.exit(1)
    else:
        opt.output_dir = os.path.join('..', "results", opt.arch, opt.exp_id)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    main(opt)