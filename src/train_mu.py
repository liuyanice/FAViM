import os, argparse, math
import numpy as np
from glob import glob
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from medpy.metric.binary import hd, dc, assd, jc

#from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter
from lib.sampling_points import sampling_points, point_sample
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import time


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='xboundformer_mu')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--net_layer', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='isic2018')#isic2016
    parser.add_argument('--exp_name', type=str, default='train')
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--lr_seg', type=float, default=1e-4)  #0.0003
    parser.add_argument('--n_epochs', type=int, default=400)  #100
    parser.add_argument('--bt_size', type=int, default=32)  #36
    parser.add_argument('--seg_loss', type=int, default=1, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--patience', type=int, default=500)  #50

    # transformer

    parser.add_argument('--im_num', type=int, default=2)
    parser.add_argument('--ex_num', type=int, default=2)
    parser.add_argument('--xbound', type=int, default=1)


    #log_dir name
    parser.add_argument('--folder_name', type=str, default='401')#Default_folder

    parse_config = parser.parse_args()
    print(parse_config)
    return parse_config


def ce_loss(pred, gt):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()


def structure_loss(pred, mask):
    """            TransFuse train loss        """
    """            Without sigmoid             """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
    
    
def bce_loss(pred, mask):
    """            TransFuse train loss        """
    """            Without sigmoid             """
  #  weit = 1 + 5 * torch.abs(
  #      F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
   # wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

  #  pred = torch.sigmoid(pred)
  #  inter = ((pred * mask) * weit).sum(dim=(2, 3))
  #  union = ((pred + mask) * weit).sum(dim=(2, 3))
 #   wiou = 1 - (inter + 1) / (union - inter + 1)
    return wbce.mean()
    
    
def dice_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()

#-------------------------- train func --------------------------#
def train(epoch):
    model.train()
    iteration = 0
    for batch_idx, batch_data in enumerate(train_loader):
        #         print(epoch, batch_idx)
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        if parse_config.arch == 'transfuse':
            lateral_map_4, lateral_map_3, lateral_map_2 = model(data)

            loss4 = structure_loss(lateral_map_4, label)
            loss3 = structure_loss(lateral_map_3, label)
            loss2 = structure_loss(lateral_map_2, label)

            loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss2.item(),
                            loss3.item(), loss4.item()))

        else:
           # P2, pixoutput = model(data)
            P2, pixoutput= model(data, label)
           # print('gt_points==',gt_points.shape)
            if parse_config.im_num + parse_config.ex_num > 0:
                seg_loss = 0.0
               # for p in P2:
              #      seg_loss = seg_loss + structure_loss(p, label)+dice_loss(p, label)
                seg_loss = seg_loss+bce_loss(pixoutput, label)+dice_loss(pixoutput, label)
                loss = seg_loss# +1*F.cross_entropy(rend, gt_points, ignore_index=255)
              #  print('loss====', seg_loss.shape)
              #  if batch_idx % 50 == 0:
               #     show_image = [label[0], F.sigmoid(pixoutput[0][0])]
               #     show_image = torch.cat(show_image, dim=2)
              #      show_image = show_image.repeat(3, 1, 1)
              #      show_image = torch.cat([data[0], show_image], dim=2)

              #      writer.add_image('pred/all',
             #                        show_image,
             #                        epoch * len(train_loader) + batch_idx,
             #                        dataformats='CHW')

            else:
                seg_loss = 0.0
              #  for p in P2:
              #      seg_loss = seg_loss + structure_loss(p, label)+dice_loss(p, label)
                seg_loss = seg_loss+bce_loss(pixoutput, label)+dice_loss(pixoutput, label)
                loss = seg_loss #+1*F.cross_entropy(rend, gt_points, ignore_index=255)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t[lateral-2: {:.4f}, lateral-3: {:0.4f}]'#, lateral-4: {:0.4f}
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss,
                            seg_loss))

    print("Iteration numbers: ", iteration)


#-------------------------- eval func --------------------------#
def evaluation(epoch, loader):
    model.eval()
    dice_value = 0
    jc_value = 0
    dice_average = 0
    jc_average = 0
   # numm = 0
    labels = []
    pres = []
    for batch_idx, batch_data in enumerate(loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        # point_All = (batch_data['point_data'] > 0).cuda().float()
        #point_All = nn.functional.max_pool2d(point_All,
        #                                 kernel_size=(16, 16),
        #                                 stride=(16, 16))

        with torch.no_grad():
            if parse_config.arch == 'transfuse':
                _, _, output = model(data)
                loss_fuse = structure_loss(output, label)
            elif parse_config.arch == 'xboundformer':
                output, pixoutput = model(
                    data)
                loss = 0
            elif parse_config.arch == 'xboundformer_mu':
                output, pixoutput = model(
                    data, label)
                loss = 0
            if parse_config.arch == 'transfuse':
                loss = loss_fuse

           # output = output.cpu().numpy() > 0.5
            pixoutput = pixoutput.cpu().numpy() > 0.5

        label = label.cpu().numpy()
        assert (pixoutput.shape == label.shape)
        labels.append(label)
        pres.append(pixoutput)
       # dice_ave1= dc(pixoutput, label)
       # dice_ave2 = dc(output, label)
     #   dice_ave  = dc(pixoutput, label)
       # iou_ave1 = jc(pixoutput, label)
       # iou_ave2 =  jc(output, label)
      #  iou_ave =  jc(pixoutput, label)
      #  dice_value += dice_ave
      #  iou_value += iou_ave
        #numm += 1
    labels = np.concatenate(labels, axis=0)
    pres = np.concatenate(pres, axis=0)
    for _id in range(labels.shape[0]):
        dice_ave = dc(labels[_id], pres[_id])
        jc_ave = jc(labels[_id], pres[_id])
        dice_value += dice_ave
        jc_value += jc_ave
    print("labels.shape[0]===",labels.shape[0])
    dice_average = dice_value / (labels.shape[0])
    jc_average = jc_value / (labels.shape[0])
  #  dice_average = dice_value / numm
  #  iou_average = iou_value / numm
    writer.add_scalar('val_metrics/val_dice', dice_average, epoch)
    writer.add_scalar('val_metrics/val_iou', jc_average, epoch)
    print("Average dc value of evaluation dataset = ", dice_average)
    print("Average jc value of evaluation dataset = ", jc_average)
    return dice_average, jc_average, loss


if __name__ == '__main__':
    #-------------------------- get args --------------------------#
    parse_config = get_cfg()

    #-------------------------- build loggers and savers --------------------------#
    exp_name = parse_config.dataset + '/' + parse_config.exp_name + '_loss_' + str(
        parse_config.seg_loss) + '_aug_' + str(
            parse_config.aug
        ) + '/' + parse_config.folder_name + '/fold_' + str(parse_config.fold)

    os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
    os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
    writer = SummaryWriter('logs/{}/log'.format(exp_name))
    save_path = 'logs/{}/model/best.pkl'.format(exp_name)
    latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)

    EPOCHS = parse_config.n_epochs
    os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
    device_ids = range(torch.cuda.device_count())

    #-------------------------- build dataloaders --------------------------#
    if parse_config.dataset == 'isic2018':
        from utils.isbi2018_new import norm01, myDataset

        dataset = myDataset(fold=parse_config.fold,
                            split='train',
                            aug=parse_config.aug)
        dataset2 = myDataset(fold=parse_config.fold, split='valid', aug=False)
    elif parse_config.dataset == 'isic2016':
        from utils.isbi2016_new import norm01, myDataset

        dataset = myDataset(split='train', aug=parse_config.aug)
        dataset2 = myDataset(split='valid', aug=False)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=parse_config.bt_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset2,
        batch_size=1,  #parse_config.bt_size
        shuffle=True,  #True
        num_workers=2,
        pin_memory=True,
        drop_last=True)  #True

    #-------------------------- build models --------------------------#
    if parse_config.arch is 'xboundformer':
        from lib.xboundformer import _segm_pvtv2
        model = _segm_pvtv2(1, parse_config.im_num, parse_config.ex_num,
                            parse_config.xbound, 384).cuda()#352
    elif parse_config.arch is 'xboundformer_mu':
        from lib.xboundformer_mu import _segm_pvtv2
        model = _segm_pvtv2(1, parse_config.im_num, parse_config.ex_num,
                            parse_config.xbound, 384).cuda()#352                      
    elif parse_config.arch == 'transfuse':
        from lib.TransFuse.TransFuse import TransFuse_S
        model = TransFuse_S(pretrained=True).cuda()

    if len(device_ids) > 1:  # 多卡训练
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)

    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    criteon = [None, ce_loss][parse_config.seg_loss]

    #-------------------------- start training --------------------------#

    max_dice = 0
    max_iou = 0
    best_ep = 0

    min_loss = 10
    min_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        start = time.time()
        train(epoch)
        dice, iou, loss = evaluation(epoch, val_loader)
        scheduler.step()

        if loss < min_loss:
            min_epoch = epoch
            min_loss = loss
        else:
            if epoch - min_epoch >= parse_config.patience:
                print('Early stopping!')
                break
        if iou > max_iou:
            max_iou = iou
            best_ep = epoch
            torch.save(model.state_dict(), save_path)
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping!')
                break
        torch.save(model.state_dict(), latest_path)
        time_elapsed = time.time() - start
        print(
            'Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))
