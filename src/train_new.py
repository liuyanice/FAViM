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
import pickle
from medpy.metric.binary import hd, dc, assd, jc

#from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter
from lib.sampling_points import sampling_points, point_sample
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import time


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='xboundformer_new')
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
    parser.add_argument('--folder_name', type=str, default='909')#Default_folder

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

def generate_consistency_masks(original_pred, enhanced_preds, threshold=0.5):
    """
    生成一致性和不一致性掩码。
    参数:
        original_pred (torch.Tensor): 原始图像的预测 [B, H, W]
        enhanced_preds (list of torch.Tensor): 增强图像的预测列表，长度为2 [B, H, W]
        threshold (float): 用于二值化预测结果的阈值
    返回:
        reliable_mask (torch.Tensor): 一致性掩码（可靠区域） [B, H, W]
        unreliable_mask (torch.Tensor): 不一致性掩码（不可靠区域） [B, H, W]
    """
    # 将预测二值化
    original_binary = (original_pred > threshold).int()  # 原始预测二值化
    enhanced_binaries = [(pred > threshold).int() for pred in enhanced_preds]  # 增强预测二值化

    # 计算可靠和不可靠掩码
    # 可靠：如果任何一个增强与原始一致
    reliable_mask = ((enhanced_binaries[0] == original_binary) | 
                     (enhanced_binaries[1] == original_binary)).int()

    # 不可靠：如果两个增强都与原始不一致
    unreliable_mask = ((enhanced_binaries[0] != original_binary) & 
                       (enhanced_binaries[1] != original_binary)).int()

    return reliable_mask, unreliable_mask
    
def generate_consistency_masks_old(original_pred, enhanced_preds, threshold=0.5):
    """
    生成一致性和不一致性掩码。
    参数:
        original_pred (torch.Tensor): 原始图像的预测 [B, H, W]
        enhanced_preds (list of torch.Tensor): 增强图像的预测列表 [B, H, W]
        threshold (float): 用于二值化预测结果的阈值
    返回:
        reliable_mask (torch.Tensor): 一致性掩码（可靠区域） [B, H, W]
        unreliable_mask (torch.Tensor): 不一致性掩码（不可靠区域） [B, H, W]
    """
    # 将预测二值化
    original_binary = (original_pred > threshold).int()  # 原始预测二值化
    enhanced_binaries = [(pred > threshold).int() for pred in enhanced_preds]  # 增强预测二值化

    # 初始化一致性计数
    agreement_count = torch.zeros_like(original_binary, dtype=torch.int32)

    # 统计一致性数量
    for enhanced_binary in enhanced_binaries:
        agreement_count += (enhanced_binary == original_binary).int()

    # 一致性掩码：大多数增强预测与原始预测一致
    reliable_mask = agreement_count >= (len(enhanced_preds) // 2 + 1)

    # 不一致性掩码：大多数增强预测与原始预测不一致
    unreliable_mask = agreement_count < (len(enhanced_preds) // 2 + 1)

    return reliable_mask, unreliable_mask
    
def selective_entropy_constraint(pred, reliable_mask, unreliable_mask, alpha=1.0, beta=1.0):
    """
    改进版选择性熵约束：对可靠像素最小化熵，对不可靠像素最大化熵。
    参数:
        pred (torch.Tensor): 模型预测概率 [B, C, H, W]
        reliable_mask (torch.Tensor): 可靠像素掩码 [B, H, W]
        unreliable_mask (torch.Tensor): 不可靠像素掩码 [B, H, W]
        alpha (float): 可靠区域熵损失的权重
        beta (float): 不可靠区域熵损失的权重
    返回:
        loss (torch.Tensor): 选择性熵约束损失
    """
    # 确保概率值在 [1e-9, 1.0] 范围内，避免数值溢出
    pred = pred.clamp(1e-9, 1.0)

    # 计算每个像素的熵
    entropy = -torch.sum(pred * torch.log(pred), dim=1)  # [B, H, W]

    # 将掩码转换为浮点型
    reliable_mask = reliable_mask.float()
    unreliable_mask = unreliable_mask.float()

    # 计算可靠区域的熵损失（最小化熵）
    reliable_entropy_loss = (entropy * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)

    # 计算不可靠区域的熵损失（最大化熵）
    unreliable_entropy_loss = -(entropy * unreliable_mask).sum() / (unreliable_mask.sum() + 1e-8)

    # 总损失，加入权重参数
    loss = alpha * reliable_entropy_loss + beta * unreliable_entropy_loss
    return loss

def extract_boundary(mask, dilation_ratio=0.02):
    """ 提取掩码的边界 """
    # 直接获取高度和宽度
  #  h, w = mask.shape[-2:]  # 获取高度和宽度
    
    # 计算对角线
 #   img_diag = torch.sqrt(torch.tensor(h ** 2 + w ** 2, device=mask.device))  # 将h和w转换为张量
  #  dilation = int(round(dilation_ratio * img_diag.item()))
    
    
    # 膨胀操作：使用最大池化进行膨胀
    dilated_mask = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1)

    # 取反操作
    inverted_mask = 1 - mask.float()  # 将原图像取反

    # 再次膨胀操作
    dilated_inverted_mask = F.max_pool2d(inverted_mask, kernel_size=3, stride=1, padding=1)
    dilated_inverted_mask = 1- dilated_inverted_mask
    # 计算边界：原始膨胀图像减去反向膨胀图像
    boundary = dilated_mask - dilated_inverted_mask

    # 将边界图像二值化
    boundary = (boundary > 0).float()  # 将边界转换为二值图像
    
    return boundary

def extract_boundary_pred(mask, dilation_ratio=0.02):
    """ 提取掩码的边界 """
     # 膨胀操作：使用最大池化进行膨胀
    mask = torch.sigmoid(mask)
    mask = (mask >0.5)
    dilated_mask = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1)

    # 取反操作
    inverted_mask = 1 - mask.float()  # 将原图像取反

    # 再次膨胀操作
    dilated_inverted_mask = F.max_pool2d(inverted_mask, kernel_size=3, stride=1, padding=1)
    dilated_inverted_mask = 1- dilated_inverted_mask
    # 计算边界：原始膨胀图像减去反向膨胀图像
    boundary = dilated_mask - dilated_inverted_mask

    # 将边界图像二值化
    boundary = (boundary > 0).float()  # 将边界转换为二值图像
    return boundary

def boundary_loss(pred_boundary, target_boundary, dilation_ratio=0.02):
    """
    基于边界的损失函数。
    
    参数:
    prediction (torch.Tensor): 预测的软概率掩码
    target (torch.Tensor): 标注的二值掩码
    dilation_ratio (float): 膨胀比率
    
    返回:
    loss (torch.Tensor): 边界损失
    """
    # 计算预测和目标的边界掩码
    pred_boundary = torch.sigmoid(pred_boundary)
   # pred_boundary = extract_boundary_pred(prediction, dilation_ratio)
  #  target_boundary = extract_boundary(target, dilation_ratio)

    # 计算边界损失
    loss = F.binary_cross_entropy(pred_boundary, target_boundary)
    return loss

def tversky(y_true, y_pred, smooth=1e-7, alpha=0.7):
    """
    Tversky loss function.
    Parameters:
    y_true (torch.Tensor): Ground truth values.
    y_pred (torch.Tensor): Predicted values.
    smooth (float): Smoothing factor.
    alpha (float): Trade-off parameter.
    Returns:
    torch.Tensor: Tversky loss.
    """
    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    tversky_score = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    return tversky_score

def tversky_loss(y_true, y_pred):
    """
    Tversky loss function.
    Parameters:
    y_true (torch.Tensor): Ground truth values.
    y_pred (torch.Tensor): Predicted values.
    Returns:
    torch.Tensor: Tversky loss.
    """
    return 1 - tversky(y_true, y_pred)

def focal_tversky(y_true, y_pred, gamma=0.75):
    """
    Focal Tversky loss function.
    Parameters:
    y_true (torch.Tensor): Ground truth values.
    y_pred (torch.Tensor): Predicted values.
    gamma (float): Focusing parameter.
    Returns:
    torch.Tensor: Focal Tversky loss.
    """
    tversky_score = tversky(y_true, y_pred)
    return torch.pow(1 - tversky_score, gamma)
    
def boundary_iou_loss(pred_mask, true_mask, dilation_ratio=0.02):
    """ 计算边界IoU损失 """
    pred_boundary = extract_boundary_pred(pred_mask, dilation_ratio)
    true_boundary = extract_boundary(true_mask, dilation_ratio)

    intersection = torch.logical_and(pred_boundary, true_boundary).sum()
    union = torch.logical_or(pred_boundary, true_boundary).sum()
    iou = intersection / union if union != 0 else torch.tensor(0.0, device=pred_mask.device)

    return 1 - iou  # 损失是1减去IoU，因为我们想最大化IoU
#-------------------------- train func --------------------------#

def bce_loss(pred, target):
    """二元交叉熵损失函数，支持logits输入"""
    batch_size = pred.size(0)
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    return F.binary_cross_entropy_with_logits(pred_flat, target_flat)

def dice1_loss(pred, target, smooth=1):
    """Dice损失函数，自动处理sigmoid转换"""
    pred = torch.sigmoid(pred)  # 将logits转换为概率
    batch_size = pred.size(0)
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()

def bce_dice_loss(pred, target, wb=5, wd=1):
    """组合的BCE+Dice损失函数"""
    return wb * bce_loss(pred, target) + wd * dice1_loss(pred, target)     
def train(epoch):
    model.train()
    iteration = 0
   # criterion = BceDiceLoss(wb=1, wd=1)
    for batch_idx, batch_data in enumerate(train_loader):
        #         print(epoch, batch_idx)
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        boundary_data = batch_data['boundary'].cuda().float()
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
           # enhanced_preds=[]
           # P2, pixoutput = model(data)
            #P1, P2, pixoutput, pixoutput_b = model(data)
            pixoutput = model(data)
           # original_pred = pixoutput.squeeze(1)
           # clahe_P2, clahe_pixoutput, clahe_xbm_feats, clahe_xbm_masks= model(clahe_data)
            #enhanced_pred1 = clahe_pixoutput.squeeze(1)
           # gamma_P2, gamma_pixoutput, gamma_xbm_feats, gamma_xbm_masks= model(gamma_data)
          #  enhanced_pred2 = gamma_pixoutput.squeeze(1) 
          #  enhanced_preds = [enhanced_pred1, enhanced_pred2]
          #  reliable_mask, unreliable_mask = generate_consistency_masks(original_pred, enhanced_preds)
           
           # selective_loss = selective_entropy_constraint(pixoutput, reliable_mask, unreliable_mask)
           # print('gt_points==',gt_points.shape)
            if parse_config.im_num + parse_config.ex_num > 0:
            #    seg_loss = 0.0
            #    b_seg_loss = 0.0
            #    for p in P1:
            #        b_seg_loss = b_seg_loss + dice_loss(p, boundary_data)
              #  for p in P2:
            #        seg_loss = seg_loss + structure_loss(p, label)+dice_loss(p, label)
             #   b_seg_loss = (b_seg_loss / len(P1))
             #   seg_loss = (seg_loss / len(P2))
              #  nll_loss = ((label - mean) ** 2 / (2 * variance + 1e-6)).mean() + torch.log(variance + 1e-6).mean()
             #   loss = seg_loss+b_seg_loss +structure_loss(pixoutput, label)+dice_loss(pixoutput, label)+dice_loss(pixoutput_b, boundary_data)#+min(epoch * 0.01, 0.5)*selective_entropy_constraint(pixoutput, reliable_mask, unreliable_mask, 1.0, 1.0)+0.1*nll_loss# +1*F.cross_entropy(rend, gt_points, ignore_index=255)
            #    loss = structure_loss(pixoutput, label)+dice_loss(pixoutput, label)
                loss =bce_dice_loss(pixoutput, label)
              #  print('loss====', seg_loss.shape)
                if batch_idx % 50 == 0:
                  #  show_image = [label[0], F.sigmoid(P2[0][0])]
                    show_image = [label[0], F.sigmoid(pixoutput[0])]
                    show_image = torch.cat(show_image, dim=2)
                    show_image = show_image.repeat(3, 1, 1)
                    show_image = torch.cat([data[0], show_image], dim=2)

                    writer.add_image('pred/all',
                                     show_image,
                                     epoch * len(train_loader) + batch_idx,
                                     dataformats='CHW')

            else:
             #   b_seg_loss = 0.0
             #   seg_loss = 0.0
             #   for p in P1:
             #       b_seg_loss = b_seg_loss + dice_loss(p, boundary_data)
             #   for p in P2:
             #       seg_loss = seg_loss + structure_loss(p, label)+dice_loss(p, label)
             #   b_seg_loss = (b_seg_loss / len(P1))
             #   seg_loss = (seg_loss / len(P2))
               # loss = seg_loss+b_seg_loss +structure_loss(pixoutput, label)+dice_loss(pixoutput, label)+dice_loss(pixoutput_b, boundary_data)#+min(epoch * 0.01, 0.5)*selective_entropy_constraint(pixoutput, reliable_mask, unreliable_mask, 1.0, 1.0)+0.1*nll_loss #+1*F.cross_entropy(rend, gt_points, ignore_index=255)
             #   loss = structure_loss(pixoutput, label)+dice_loss(pixoutput, label)
                loss =bce_dice_loss(pixoutput, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t[loss: {:.4f}]'#, lateral-4: {:0.4f} , seg_loss: {:0.4f}, b_loss: {:0.4f}
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss))

    print("Iteration numbers: ", iteration)

def save_xbm_states(xbm1, xbm2, save_path):
    """将 XBM1 和 XBM2 的状态保存到文件"""
    with open(save_path, 'wb') as f:
        pickle.dump({
            'xbm1': xbm1,
            'xbm2': xbm2
        }, f)
        
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
            elif parse_config.arch == 'xboundformer_new':
              #  P1, P2, pixoutput, pixoutput_b = model(data)
                pixoutput = model(data)
                loss = 0
            if parse_config.arch == 'transfuse':
                loss = loss_fuse
            
           # output = output.cpu().numpy() > 0.5
            pixoutput = torch.sigmoid(pixoutput)
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
    save_path1 = 'logs/{}/model/xbm_states.pkl'.format(exp_name)
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
        shuffle=False,  #True
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
    elif parse_config.arch is 'xboundformer_new':
        from lib.xboundformer_new import _segm_pvtv3
        model = _segm_pvtv3(1, parse_config.im_num, parse_config.ex_num,
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
        if dice > max_dice:
            max_dice = dice
            best_ep = epoch
            torch.save(model.state_dict(), save_path)
           # save_xbm_states(xbm_feats, xbm_masks, save_path1)
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping!')
                break
        torch.save(model.state_dict(), latest_path)
        time_elapsed = time.time() - start
        print(
            'Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))
