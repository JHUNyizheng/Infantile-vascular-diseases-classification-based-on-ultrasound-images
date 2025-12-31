from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
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
from thop import profile
from src.util.data_parallel import DataParallel

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


import matplotlib.pyplot as plt

# 相关库

def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm = confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    # 显示
    plt.show()



# def col_pic():
#     for i in range(10):
#         y_label = []
#         y_pred = []
#         with open("pre_lab_" + str(i) + ".csv") as f:
#             f1 = csv.reader(f)
#             for line in f1:
#                 y_label.append(int(float(line[0])))
#                 # if float(line[1]) > 0.5:
#                 #     y_pred.append(1)
#                 # else:
#                 #     y_pred.append(0)
#                 y_pred.append(float(line[1]))
#             ro_curve(y_pred,y_label,"auc_val_1","Fold" + str(i+1))

# def ro_curve(y_pred, y_label, figure_file, method_name):
#     '''
#         y_pred is a list of length n.  (0,1)
#         y_label is a list of same length. 0/1
#         https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
#     '''
#     y_label = np.array(y_label)
#     y_pred = np.array(y_pred)
#     # fpr = dict()
#     # tpr = dict()
#     # roc_auc = dict()
#     # fpr[0], tpr[0], _ = precision_recall_curve(y_label, y_pred)
#     # roc_auc[0] = auc(fpr[0], tpr[0])
#     # lw = 2
#     # plt.plot(fpr[0], tpr[0],
#     #      lw=lw, label= method_name + ' (area = %0.2f)' % roc_auc[0])
#     # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     # plt.xlim([0.0, 1.0])
#     # plt.ylim([0.0, 1.05])
#     # fontsize = 14
#     # plt.xlabel('Recall', fontsize = fontsize)
#     # plt.ylabel('Precision', fontsize = fontsize)
#     # plt.title('Precision Recall Curve')
#     # plt.legend(loc="lower right")
#     # plt.savefig(figure_file)
#     lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)
# #   plt.plot([0,1], [no_skill, no_skill], linestyle='--')
#     plt.plot(lr_recall, lr_precision, lw = 2, label= method_name + ' (area = %0.2f)' % average_precision_score(y_label, y_pred))
#     fontsize = 14
#     plt.xlabel('Recall', fontsize = fontsize)
#     plt.ylabel('Precision', fontsize = fontsize)
#     plt.title('Precision Recall Curve')
#     plt.legend()
#     plt.savefig(figure_file)
#     return

def train(model, train_loader, vaild_loader, optimizer, scaler, scheduler, epoch, logger, opt):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    class_precision_meter = AverageMeter()
    class_recall_meter = AverageMeter()
    class_f1_meter = AverageMeter()

    box_precision_meter = AverageMeter()
    box_recall_meter = AverageMeter()
    mAP50_meter = AverageMeter()
    mAP50_95_meter = AverageMeter()
    fitness_meter = AverageMeter()


    end = time.time()
    time1 = time.time()
    classes_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    detect_criterion = v8DetectionLoss(model)
    # keypoint_loss = KeypointLoss()
    keypoint_loss = nn.MSELoss()
    # model.train()
    # train_len = len(train_loader)
    validator = DetectionValidator(model)
    # # for crops_a, coords, flags, crops_RGB in train_loader:
    # for i, (image_gray, image_color, gray_coords, phlebolith_labels, classes) in enumerate(train_loader):
    #     image_gray = image_gray.cuda(non_blocking=True).type(torch.float32)
    #     image_color = image_color.cuda(non_blocking=True).type(torch.float32)
    #     gray_coords = gray_coords.cuda(non_blocking=True).type(torch.float32)
    #     # phlebolith_coords = phlebolith_coords.cuda(non_blocking=True).type(torch.float32)
    #     classes = classes.cuda(non_blocking=True).type(torch.float32)
    #
    #     with torch.cuda.amp.autocast(scaler is not None):
    #         box_pred, class_pred = model(image_gray, image_color)
    #
    #     loss_class = classes_criterion(class_pred.type(torch.float32), classes.squeeze(-1).type(torch.int64))
    #
    #     if 1 in phlebolith_labels['cls']:
    #         # compute output and loss
    #         loss_box, loss_items = detect_criterion(box_pred, phlebolith_labels)
    #         loss = loss_class + loss_box
    #     else:
    #         # continue
    #         loss_box = 0
    #         loss = loss_class
    #     #
    #     # if 1 in phlebolith_labels['cls']:
    #     #     image_gray = image_gray.cuda(non_blocking=True).type(torch.float32)
    #     #     image_color = image_color.cuda(non_blocking=True).type(torch.float32)
    #     #     gray_coords = gray_coords.cuda(non_blocking=True).type(torch.float32)
    #     #     # phlebolith_coords = phlebolith_coords.cuda(non_blocking=True).type(torch.float32)
    #     #     classes = classes.cuda(non_blocking=True).type(torch.float32)
    #     #     # compute output and loss
    #     #     with torch.cuda.amp.autocast(scaler is not None):
    #     #         box_pred, class_pred = model(image_gray, image_color)
    #     #     # loss_class = classes_criterion(class_pred.type(torch.float32), classes.squeeze(-1).type(torch.int64))
    #     #     loss_box, loss_items = detect_criterion(box_pred, phlebolith_labels)
    #     #     loss = loss_box
    #     #     loss_class = 0
    #     # else:
    #     #     continue
    #
    #
    #     # if opt.step == '1':
    #     #     # compute output and loss
    #     #     with torch.cuda.amp.autocast(scaler is not None):
    #     #         box_pred, class_pred = model(image_gray, image_color)
    #     #     loss_class = classes_criterion(class_pred.type(torch.float32), classes.squeeze(-1).type(torch.int64))
    #     #     loss_box = 0
    #     #     loss = loss_class
    #     # elif opt.step == '2':
    #     #     if 1 in phlebolith_labels['cls']:
    #     #         # compute output and loss
    #     #         with torch.cuda.amp.autocast(scaler is not None):
    #     #             box_pred, class_pred = model(image_gray, image_color)
    #     #         loss_box, loss_items = detect_criterion(box_pred, phlebolith_labels)
    #     #         loss_class = 0
    #     #         loss = loss_box
    #     #     else:
    #     #         continue
    #
    #     if torch.isnan(loss):
    #         ValueError("Loss is NaN")
    #
    #         # loss = loss.mean()
    #     optimizer.zero_grad()
    #
    #     if opt.fp16:
    #         scaler.scale(loss).backward()
    #         scaler.unscale_(optimizer)
    #         scaler.step(optimizer)
    #         scaler.update()
    #     else:
    #         loss.backward()
    #         optimizer.step()
    #     scheduler.step()
    #
    #     # avg loss from batch size
    #     loss_meter.update(loss.item(), image_gray[0].size(0))
    #     # measure elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()
    #
    #     # masks = masks.argmax(dim=1)
    #     # # miou = iou_mean(pred, label.type(torch.int64), opt.n_classes)
    #     # result = eval_semantic_segmentation(masks.detach().cpu().numpy(), segm.type(torch.int64).detach().cpu().numpy())
    #     #
    #     # valid_idx = segm != -1
    #     # acc = (masks[valid_idx] == segm[valid_idx]).sum() / masks[valid_idx].numel()
    #
    #     if i % 2 == 0:
    #         lr = optimizer.param_groups[0]['lr']
    #         etas = batch_time.avg * (train_len - i)
    #         logger.info(
    #             f'Train: [{epoch}/{opt.num_epochs}][{i}/{train_len}]  '
    #             f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.4f}  '
    #             f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
    #             f'loss {loss_meter.val:.5f} ({loss_meter.avg:.5f})  '
    #             f'loss_box {loss_box:.5f} '
    #             f'loss_class {loss_class:.5f}  '
    #             # f'miou {result["miou"]:.5f}  '
    #             # f'acc {acc:.5f}  '
    #         )

    """验证"""
    model.eval()
    logger.info(
        f'-------------------------------\n'
    )
    pred_scores_list, pred_list, label_list = [],[],[]
    with torch.no_grad():
        for i, (image_gray, image_color, gray_coords, phlebolith_labels, classes) in enumerate(vaild_loader):
            # if i>3:
            #     break

            # plt.imshow(np.transpose(image_gray[0],(1, 2, 0)))
            # plt.show()
            # plt.imshow(np.transpose(image_color[0], (1, 2, 0)))
            # plt.show()

            image_color = torch.zeros_like(image_color)

            image_gray = image_gray.cuda(non_blocking=True).type(torch.float32)
            image_color = image_color.cuda(non_blocking=True).type(torch.float32)
            gray_coords = gray_coords.cuda(non_blocking=True).type(torch.float32)



            # phlebolith_coords = phlebolith_coords.cuda(non_blocking=True).type(torch.float32)
            classes = classes.cuda(non_blocking=True).type(torch.float32)

            # compute output and loss
            with torch.cuda.amp.autocast(scaler is not None):
                box_pred, class_pred = model(image_gray, image_color)

            phlebolith_labels = validator.preprocess(phlebolith_labels)

            box_pred = validator.postprocess(box_pred)
            validator.update_metrics(box_pred, phlebolith_labels)
            stats = validator.get_stats()
            validator.finalize_metrics()

            precision = stats['metrics/precision(B)']
            recall = stats['metrics/recall(B)']
            mAP50 = stats['metrics/mAP50(B)']
            mAP50_95 = stats['metrics/mAP50-95(B)']
            fitness = stats['fitness']

            pred_scores_list.append(class_pred[0].detach().cpu().numpy())

            class_pred = class_pred.argmax(dim=1)

            sklearn_precision = precision_score(classes.detach().cpu().numpy(), class_pred.detach().cpu().numpy(), average='macro')
            sklearn_recall = recall_score(classes.detach().cpu().numpy(), class_pred.detach().cpu().numpy(), average='macro')
            sklearn_f1 = f1_score(classes.detach().cpu().numpy(), class_pred.detach().cpu().numpy(), average='macro')

            pred_list.append(class_pred[0].detach().cpu().numpy())
            label_list.append(classes[0][0].detach().cpu().numpy())

            # 分类任务
            class_precision_meter.update(sklearn_precision, image_gray[0].size(0))
            class_recall_meter.update(sklearn_recall, image_gray[0].size(0))
            class_f1_meter.update(sklearn_f1, image_gray[0].size(0))
            # 静脉瘤检测任务
            box_precision_meter.update(precision, image_gray[0].size(0))
            box_recall_meter.update(recall, image_gray[0].size(0))
            mAP50_meter.update(mAP50, image_gray[0].size(0))
            mAP50_95_meter.update(mAP50_95, image_gray[0].size(0))
            fitness_meter.update(fitness, image_gray[0].size(0))

            # validator.plot_val_samples(box_pred, i)

            if i % 20 == 0:
                logger.info(
                    # f'Train: [{epoch}/{opt.num_epochs}][{i}/{train_len}]  '
                    # f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.4f}  '
                    f'step: {i:.0f} time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                    f' | \n'
                    f'class_precision_meter {class_precision_meter.val:.5f} ({class_precision_meter.avg:.5f}) '
                    f'class_recall_meter {class_recall_meter.val:.5f} ({class_recall_meter.avg:.5f}) '
                    f'class_f1_meter {class_f1_meter.val:.5f} ({class_f1_meter.avg:.5f}) '
                    f' | \n'
                    f'box_precision_meter {box_precision_meter.val:.5f} ({box_precision_meter.avg:.5f}) '
                    f'box_recall_meter {box_recall_meter.val:.5f} ({box_recall_meter.avg:.5f}) '
                    f'mAP50_meter {mAP50_meter.val:.5f} ({mAP50_meter.avg:.5f}) '
                    f'mAP50_95_meter {mAP50_95_meter.val:.5f} ({mAP50_95_meter.avg:.5f}) '
                    f'fitness_meter {fitness_meter.val:.5f} ({fitness_meter.avg:.5f}) '
                )

                logger.info(
                    f'-------------------------------\n')



        # pred_list = np.reshape(pred_list,len(pred_list))
        # label_list = np.reshape(label_list, len(label_list))
        print('precision_score:', precision_score(pred_list, label_list, average='macro'))
        print('recall_score:', recall_score(pred_list, label_list, average='macro'))
        print('f1_score:', f1_score(pred_list, label_list, average='macro'))
        print(confusion_matrix(label_list,pred_list))

        class_name = ['NR', 'VM', 'IH']
        plot_matrix(label_list, pred_list, [0, 1, 2], title='Confusion Matrix for IVDNet',
                    axis_labels=class_name)

        n_classes = 3
        true_labels_binarized = label_binarize(label_list, classes=[0, 1, 2])
        # 绘制每个类别的PR曲线
        plt.figure(figsize=(10, 8))
        pred_scores_list = np.array(pred_scores_list)
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(true_labels_binarized[:, i], pred_scores_list[:, i])
            average_precision = average_precision_score(true_labels_binarized[:, i], pred_scores_list[:, i])
            plt.plot(recall, precision, marker='.', label=f'{class_name[i]} (AP = {average_precision:.2f})')
        # 设置图形的标题和标签
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for IVDNet')
        plt.legend()
        plt.grid()
        plt.show()

        # 绘制每个类别的ROC曲线
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(true_labels_binarized[:, i], pred_scores_list[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, marker='.', label=f'{class_name[i]} (AUC = {roc_auc:.2f})')

        # 设置图形的标题和标签
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for IVDNet')
        plt.legend()
        plt.grid()
        plt.show()






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

    opt.num_instances = len(train_loader.dataset)
    opt.num_instances_val = len(vaild_loader.dataset)
    logger.info(f"length of training dataset: {opt.num_instances}")
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
    start_epoch = 0
    """ Training the model """
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        opt.now_epcho = epoch
        train_sampler.set_epoch(epoch)
        # start_time = time.time()
        train(model, train_loader, vaild_loader, optimizer, scaler, scheduler, epoch, logger, opt)

        save_checkpoint(logger, opt, epoch, model, optimizer, scheduler, scaler)

        # if dist.get_rank() == 0 and (epoch % int(opt.save_freq) == 0 or epoch in opt.save_point):
        #     save_checkpoint(logger, opt, epoch, model, optimizer, scheduler, scaler)

        if dist.get_rank() == 0 and epoch % int(opt.save_freq) == 0:
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
            file_name = os.path.join(opt.output_dir, f'ckpt_epoch_{epoch}.pth')
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

