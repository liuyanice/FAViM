import os, argparse, sys, tqdm, logging, cv2
import torch
import torch.nn as nn
from thop import profile
import numpy as np
import time
import numba
from time import *
from glob import glob
import torch.nn.functional as F
from medpy.metric.binary import hd, hd95, dc, jc, assd, sensitivity, specificity #, accuracy
#from lib.xboundformer_mu import _segm_pvtv2
from lib.xboundformer_new import _segm_pvtv3
from scipy.ndimage import morphology
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import convolve
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='xboundformer_new')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--net_layer', type=int, default=50)
parser.add_argument('--dataset', type=str, default='isic2018')
parser.add_argument('--exp_name', type=str, default='test')
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
parser.add_argument('--folder_name', type=str, default='409')#Default_folder

parse_config = parser.parse_args()
print(parse_config)
os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if parse_config.dataset == 'isic2018':
    from utils.isbi2018_new import norm01, myDataset
    dataset = myDataset(parse_config.fold, 'test', aug=False)#valid
elif parse_config.dataset == 'isic2016':
    from utils.isbi2016_new import norm01, myDataset
    dataset = myDataset('valid', aug=False)

if parse_config.arch == 'BAT':
    if parse_config.trans == 1:
        from lib.xboundformer import _segm_pvtv2
        model = _segm_pvtv2(1, 1, 1, 1, 352).to(device)
    else:
        from Ours.base import DeepLabV3
        model = DeepLabV3(1, parse_config.net_layer).cuda()

def Object(pred, gt):
    x = np.mean(pred[gt == 1])
    sigma_x = np.std(pred[gt == 1])
    score = 2.0 * x / (x ** 2 + 1 + sigma_x + np.finfo(np.float64).eps)

    return score
    
def fspecial_gauss(size, sigma):
       """Function to mimic the 'fspecial' gaussian MATLAB function
       """
       x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
       g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
       return g/g.sum()
       
def S_Object(pred, gt):
    pred_fg = pred.copy()
    pred_fg[gt != 1] = 0.0
    O_fg = Object(pred_fg, gt)
    
    pred_bg = (1 - pred.copy())
    pred_bg[gt == 1] = 0.0
    O_bg = Object(pred_bg, 1-gt)

    u = np.mean(gt)
    Q = u * O_fg + (1 - u) * O_bg

    return Q
    
def MAE(pred, gt):
    Q = np.mean(np.abs(gt - pred))

    return Q

def centroid(gt):
    if np.sum(gt) == 0:
        return gt.shape[0] // 2, gt.shape[1] // 2
    
    else:
        x, y = np.where(gt == 1)
        return int(np.mean(x).round()), int(np.mean(y).round())

def divide(gt, x, y):
    LT = gt[:x, :y]
    RT = gt[x:, :y]
    LB = gt[:x, y:]
    RB = gt[x:, y:]

    w1 = LT.size / gt.size
    w2 = RT.size / gt.size
    w3 = LB.size / gt.size
    w4 = RB.size / gt.size

    return LT, RT, LB, RB, w1, w2, w3, w4

def ssim(pred, gt):
    x = np.mean(pred)
    y = np.mean(gt)
    N = pred.size

    sigma_x2 = np.sum((pred - x) ** 2 / (N - 1 + np.finfo(np.float64).eps))
    sigma_y2 = np.sum((gt - y) ** 2 / (N - 1 + np.finfo(np.float64).eps))

    sigma_xy = np.sum((pred - x) * (gt - y) / (N - 1 + np.finfo(np.float64).eps))

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + np.finfo(np.float64).eps)
    elif alpha == 0 and beta == 0:
        Q = 1
    else:
        Q = 0
    
    return Q

def S_Region(pred, gt):
    x, y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divide(gt, x, y)
    pred1, pred2, pred3, pred4, _, _, _, _ = divide(pred, x, y)

    Q1 = ssim(pred1, gt1)
    Q2 = ssim(pred2, gt2)
    Q3 = ssim(pred3, gt3)
    Q4 = ssim(pred4, gt4)

    Q = Q1 * w1 + Q2 * w2 + Q3 * w3 + Q4 * w4

    return Q
    
def original_WFb(pred, gt):
    E = np.abs(pred - gt)
    dst, idst = distance_transform_edt(1 - gt, return_indices=True)

    K = fspecial_gauss(7, 5)
    Et = E.copy()
    Et[gt != 1] = Et[idst[:, gt != 1][0], idst[:, gt != 1][1]]
    EA = convolve(Et, K, mode='nearest')
    MIN_E_EA = E.copy()
    MIN_E_EA[(gt == 1) & (EA < E)] = EA[(gt == 1) & (EA < E)]

    B = np.ones_like(gt)
    B[gt != 1] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5 * dst[gt != 1])
    Ew = MIN_E_EA * B

    TPw = np.sum(gt) - np.sum(Ew[gt == 1])
    FPw = np.sum(Ew[gt != 1])

    R = 1 - np.mean(Ew[gt == 1])
    P = TPw / (TPw + FPw + np.finfo(np.float64).eps)
    Q = 2 * R * P / (R + P + np.finfo(np.float64).eps)

    return Q

def StructureMeasure(pred, gt):
    y = np.mean(gt)

    if y == 0:
        x = np.mean(pred)
        Q = 1 - x
    elif y == 1:
        x = np.mean(pred)
        Q = x
    else:
        alpha = 0.5
        Q = alpha * S_Object(pred, gt) + (1 - alpha) * S_Region(pred, gt)
        if Q < 0:
            Q = 0
    
    return Q
    
def Fmeasure_calu(pred, gt, threshold):
    if threshold > 1:
        threshold = 1

    Label3 = np.zeros_like(gt)
    Label3[pred >= threshold] = 1

    NumRec = np.sum(Label3 == 1)
    NumNoRec = np.sum(Label3 == 0)

    LabelAnd = (Label3 == 1) & (gt == 1)
    NumAnd = np.sum(LabelAnd == 1)
    num_obj = np.sum(gt)
    num_pred = np.sum(Label3)

    FN = num_obj - NumAnd
    FP = NumRec - NumAnd
    TN = NumNoRec - FN

    if NumAnd == 0:
        PreFtem = 0
        RecallFtem = 0
        FmeasureF = 0
        Dice = 0
        SpecifTem = 0
        IoU = 0

    else:
        IoU = NumAnd / (FN + NumRec)
        PreFtem = NumAnd / NumRec
        RecallFtem = NumAnd / num_obj
        SpecifTem = TN / (TN + FP)
        Dice = 2 * NumAnd / (num_obj + num_pred)
        FmeasureF = ((2.0 * PreFtem * RecallFtem) / (PreFtem + RecallFtem))
    
    return PreFtem, RecallFtem, SpecifTem, Dice, FmeasureF, IoU

def AlignmentTerm(pred, gt):
    mu_pred = np.mean(pred)
    mu_gt = np.mean(gt)

    align_pred = pred - mu_pred
    align_gt = gt - mu_gt

    align_mat = 2 * (align_gt * align_pred) / (align_gt ** 2 + align_pred ** 2 + np.finfo(np.float64).eps)
    
    return align_mat

def EnhancedAlighmentTerm(align_mat):
    enhanced = ((align_mat + 1) ** 2) / 4
    return enhanced
    
def EnhancedMeasure(pred, gt):
    if np.sum(gt) == 0:
        enhanced_mat = 1 - pred
    elif np.sum(1 - gt) == 0:
        enhanced_mat = pred.copy()
    else:
        align_mat = AlignmentTerm(pred, gt)
        enhanced_mat = EnhancedAlighmentTerm(align_mat)
    
    score = np.sum(enhanced_mat) / (gt.size - 1 + np.finfo(np.float64).eps)
    return score
    
def EnhancedMeasureMax(pred, gt):
    if np.sum(gt) == 0:
        enhanced_mat = 1 - pred
    elif np.sum(1 - gt) == 0:
        enhanced_mat = pred.copy()
    else:
        align_mat = AlignmentTerm(pred, gt)
        enhanced_mat = EnhancedAlighmentTerm(align_mat)
    
    score = np.max(enhanced_mat) 
    return score
def assd2(label, pred):
    # 确保 label 和 pred 是二值化的
    label = (label > 0.5).astype(int)
    pred = (pred > 0.5).astype(int)
    
    # 计算边界点
    label_boundary = morphology.binary_erosion(label) ^ label
    pred_boundary = morphology.binary_erosion(pred) ^ pred
    
    # 计算每个边界点到另一个边界的最小距离
    label_to_pred = _one_way_distance_opt(label_boundary, pred_boundary)
    pred_to_label = _one_way_distance_opt(pred_boundary, label_boundary)
    
    # 计算平均对称曲面距离
    return (np.mean(label_to_pred) + np.mean(pred_to_label)) / 2
    
@numba.jit(nopython=True)
def _one_way_distance_opt(src_boundary, tar_boundary):
    distances = np.empty(src_boundary.sum())
    idx = 0
    for i, j in zip(*np.nonzero(src_boundary)):
        min_distance = np.inf
        for x, y in zip(*np.nonzero(tar_boundary)):
            distance = np.sqrt((i-x)**2 + (j-y)**2)
            if distance < min_distance:
                min_distance = distance
        distances[idx] = min_distance
        idx += 1
    return distances
#model = load_model(model, r'C:\Users\liuyan\Desktop\skin disease\segemanation\BA-Transformer-main\logs\isic2018\_1_1_0_e6_loss_0_aug_1\fold_0'+ '/model/best.pkl')

def apply_transparent_overlay(base_img, mask, color, alpha=0.5):
    """
    在原始图像上仅对掩码区域叠加透明颜色，同时保持其他部分的原始亮度。
    base_img: 原始图像 (彩色图像)
    mask: 掩码 (二值图，标记需要叠加的区域)
    color: 叠加颜色 (BGR 格式)
    alpha: 透明度 (0: 完全透明, 1: 不透明)
    """
    # 创建一个与 base_img 尺寸相同的彩色叠加图层
    overlay = np.zeros_like(base_img, dtype=np.uint8)
    overlay[mask > 0] = color  # 在掩码区域应用颜色

    # 仅对掩码区域进行透明叠加
    result = base_img.copy()  # 保持原图亮度
    mask_indices = mask > 0  # 掩码区域的布尔索引
    result[mask_indices] = (
        base_img[mask_indices] * (1 - alpha) + overlay[mask_indices] * alpha
    ).astype(np.uint8)

    return result  
#from utils.isbi2018_new import norm01, myDataset
#dataset = myDataset(fold='0',split='valid', aug=False)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)


def test():
    model = _segm_pvtv3(1, 2, 2, 1, 384).to(device)
    model.eval()
    num = 0

    dice_value = 0
    jc_value = 0
    acc_value = 0
    sp_value = 0
    se_value = 0
    hd95_value = 0
    assd_value = 0
    f_value = 0
    s_value = 0
    e_value = 0
    emax_value = 0
    mae_value = 0
    from tqdm import tqdm
    labels = []
    pres = []
    ##把测试集前20帧特征和labels加载进去内存
    count = 0
    
    
    for batch_idx, batch_data in tqdm(enumerate(test_loader)):
        model.load_state_dict(
            torch.load(
                r'/lab/ly/YanLiu/BUsegNew/BUsegNew/src/logs/isic2018/train_loss_1_aug_1/409/fold_0/model/best.pkl'
            ))
        
        data = batch_data['image'].to(device).float()
       # print("sssss",data.shape)
        label = batch_data['label'].to(device).float()
        with torch.no_grad():
        #    flops, params = profile(model, inputs=(data, ))
       #     print("FLOPs=", str(flops/1e9) + '{}'.format("G")) 
      #      print("params=", str(params/1e6) + '{}'.format("M"))
            t0 = time()
            pixoutput= model(data)
            print(time()-t0)
            total = sum([param.nelement() for param in model.parameters()])
 
            print("Number of parameter: %.2fM" % (total/1e6))
            #output = torch.sigmoid(output)
            #output = output[0][0]
            v=torch.sigmoid(pixoutput)
            o = v[0][0]
            o = (o.cpu().detach().numpy() >0.5).astype('uint8')*255
            print("o==", o.shape)
        #    d = (data[0,:,:,:]).cpu().detach().numpy().transpose(1, 2, 0) #8很好
          #  d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB) 
         #   print("d", d.shape)
        #    img = (((d - np.min(d))/(np.max(d)-np.min(d)))*255).astype(np.uint8)
        #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            print("data==", data.shape) 
            img = data.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            img = ((img)*255).astype(np.uint8)            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            g = (label[0,:,:,:]).cpu().detach().numpy().transpose(1, 2, 0) #8很好
           # g = (label).cpu().detach().numpy().transpose(1, 2, 0) #8很好
            img1 = (((g - np.min(g))/(np.max(g)-np.min(g)))*255).astype(np.uint8)
            img3 = img1
            img1 =np.squeeze(img1)
            print("img==", img.shape)
            print("img1==", img1.shape)
             # 确保预测（o_binary）和标签（img1）形状一致
            if len(img1.shape) == 3:  # 如果 img1 是三通道，转换为单通道
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(o.shape) == 3:  # 如果预测是三通道，转换为单通道
                o = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
            
             # 创建最终输出的彩色图像
            final_overlay = np.zeros_like(img, dtype=np.uint8)

            # 重叠区域（黄色）：预测和地面实况都为前景
            intersection = cv2.bitwise_and(o, img1)  # 找到重叠区域
            final_overlay[intersection > 0] = (0, 255, 255)  # 黄色

            # 预测独有的区域（绿色）：预测为前景但地面实况为背景
            o_only = cv2.bitwise_and(o, cv2.bitwise_not(img1))
            final_overlay[o_only > 0] = (0, 255, 0)  # 绿色

            # 地面实况独有的区域（红色）：地面实况为前景但预测为背景
            img1_only = cv2.bitwise_and(img1, cv2.bitwise_not(o))
            final_overlay[img1_only > 0] = (0, 0, 255)  # 红色
            red_mask = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
            red_mask[img1 > 0] = (0, 0, 255)
         #   cv2.imwrite(f"/lab/ly/YanLiu/BUsegNew/BUsegNew/out/{count}.png", red_mask)# label
            cv2.imwrite(f"/lab/ly/YanLiu/BUsegNew/BUsegNew/out1/{count}.png", final_overlay)# pred
          #  cv2.imwrite(f"/lab/ly/YanLiu/BUsegNew/BUsegNew/out4/{count}.png", img)# img
           # cv2.imwrite(f"/lab/ly/YanLiu/BUsegNew/BUsegNew/out2/{count}.png", o)# pred
           # cv2.imwrite(f"/lab/ly/YanLiu/BUsegNew/BUsegNew/out3/{count}.png", img3)#GT
            count =count+1
            output = pixoutput.cpu().numpy() > 0.5
            
        label = label.cpu().numpy()
       # print("11",output.shape)
      #  print("22",label.shape)
        assert (output.shape == label.shape)
        labels.append(label)
        pres.append(output)
    labels = np.concatenate(labels, axis=0)
    pres = np.concatenate(pres, axis=0)
    print("pres===",pres.shape)
    print(labels.shape, pres.shape)
    for _id in range(labels.shape[0]):
        
        dice_ave = dc(labels[_id], pres[_id])
        jc_ave = jc(labels[_id], pres[_id]) #sensitivity, specificity, accuracy
        acc_ave = (pres[_id] == labels[_id]).mean().item()
        sp_ave = specificity(labels[_id], pres[_id])
        se_ave = sensitivity(labels[_id], pres[_id])
        
      #  print(pres[_id].squeeze(0).shape)
      #  print(labels[_id].squeeze(0).shape)
        s_ave = StructureMeasure(pres[_id].squeeze(0), labels[_id].squeeze(0))
        f_ave = original_WFb(pres[_id].squeeze(0), labels[_id].squeeze(0))
        e_ave = EnhancedMeasure(pres[_id].squeeze(0), labels[_id].squeeze(0))
        emax_ave =EnhancedMeasureMax(pres[_id].squeeze(0), labels[_id].squeeze(0))
        mae_ave =MAE(pres[_id].squeeze(0), labels[_id].squeeze(0))
        try:
            hd95_ave = hd95(labels[_id].squeeze(0), pres[_id].squeeze(0))
            assd_ave = assd2(labels[_id].squeeze(0), pres[_id].squeeze(0))
        except RuntimeError:
            num += 1
            hd95_ave = 0
            assd_ave = 0
    #        f_ave = 0
        dice_value += dice_ave
        jc_value += jc_ave
        acc_value += acc_ave
        sp_value += sp_ave
        se_value += se_ave
        hd95_value += hd95_ave
        assd_value += assd_ave
   #     f_value += f_ave
        s_value += s_ave
        e_value += e_ave
        emax_value += emax_ave
        mae_value += mae_ave
    print("num===",num)
    print("labels.shape[0]===",labels.shape[0])
    dice_average = dice_value / (labels.shape[0])
    jc_average = jc_value / (labels.shape[0])
    acc_average = acc_value / (labels.shape[0])
    sp_average = sp_value / (labels.shape[0])
    se_average = se_value / (labels.shape[0])
    hd95_average = hd95_value / (labels.shape[0])
    assd_average = assd_value / (labels.shape[0])
 #   f_average = f_value / (labels.shape[0])
    s_average = s_value / (labels.shape[0])
    e_average = e_value / (labels.shape[0])
    emax_average = emax_value / (labels.shape[0])
    mae_average = mae_value / (labels.shape[0])
    logging.info('Dice value of test dataset  : %f' % (dice_average))
    logging.info('Jc value of test dataset  : %f' % (jc_average))
    logging.info('Acc value of test dataset  : %f' % (acc_average))
    logging.info('Sp value of test dataset  : %f' % (sp_average))
    logging.info('Se value of test dataset  : %f' % (se_average))
    logging.info('Hd95 value of test dataset  : %f' % (hd95_average))
    logging.info('Assd value of test dataset  : %f' % (assd_average))
  #  logging.info('f value of test dataset  : %f' % (f_average))
    logging.info('s value of test dataset  : %f' % (s_average))
    logging.info('e value of test dataset  : %f' % (e_average))
    logging.info('emax value of test dataset  : %f' % (emax_average))
    logging.info('mae value of test dataset  : %f' % (mae_average))
    print("Average dice value of evaluation dataset = ", dice_average)
    print("Average jc value of evaluation dataset = ", jc_average)
    print("Average Acc value of evaluation dataset = ", acc_average)
    print("Average Sp value of evaluation dataset = ", sp_average)
    print("Average Se value of evaluation dataset = ", se_average)
    print("Average hd95 value of evaluation dataset = ", hd95_average)
    print("Average assd value of evaluation dataset = ", assd_average)
  #  print("Average f value of evaluation dataset = ", f_average)
    print("Average s value of evaluation dataset = ", s_average)
    print("Average e value of evaluation dataset = ", e_average)
    print("Average emax value of evaluation dataset = ", emax_average)
    print("Average mae value of evaluation dataset = ", mae_average)
    return dice_average


if __name__ == '__main__':
    test()
