"""
工具函数：前向传播、AUC计算、随机种子设置
"""

import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import DatasetFolder, ImageFolder
import torchvision.transforms as transforms
from functools import partial
import scipy.io as scio
from sklearn.metrics import roc_auc_score  # ROC-AUC计算
import torch.nn.functional as F
from torchvision.utils import save_image


def forward(model, data):
    """
    前向传播预处理：将数据移至GPU并缩放
    参数:
        model: VAE模型
        data: 输入数据
    返回:
        (模型输出, 处理后的数据)
    """
    data = data.cuda().float()  # 移至GPU并转换为浮点数
    data = data * 10            # 数据缩放（增强信号）

    return model(data), data


def auc(x, label, y):
    """
    计算多种异常评分的ROC-AUC
    基于重建误差的不同度量方法
    参数:
        x: 原始图像 [batch, 1, H, W]
        label: 标签 (0=异常, 1=正常)
        y: 重建图像 [batch, 1, H, W]
    返回:
        mae_auc: 平均绝对误差的AUC
        mse_auc: 均方误差的AUC
        xujing_auc: 相对误差的AUC
        attn_auc: 注意力分数的AUC
    """
    diff = x - y  # 重建误差

    # 1. MAE（平均绝对误差）评分
    mae = diff.abs().sum(dim=[1,2,3]).detach().cpu()  # 逐像素绝对误差求和
    mae_auc = roc_auc_score(label, -mae)  # 使用负MAE（误差越小越正常）

    # 2. MSE（均方误差）评分
    mse = diff.abs().pow(2).sum(dim=[1,2,3]).detach().cpu()  # 逐像素平方误差求和
    mse_auc = roc_auc_score(label, -mse)  # 使用负MSE

    # TODO adjust thres, kernel, lambda

    # 3. 相对误差评分（xujing：基于相对误差的度量）
    xujing = (diff / x).abs().sum(dim=[1,2,3]).detach().cpu()  # 相对误差绝对值求和
    xujing_auc = roc_auc_score(label, -xujing)  # 使用负相对误差

    # 4. 注意力评分（结合空间信息）
    thres = 0.05  # 阈值，用于区分背景和信号区域
    kernel = 3    # 池化核大小
    # 创建掩码：通过最大池化找到低能量区域（背景）
    mask = torch.nn.functional.max_pool2d(
            x, kernel_size=kernel, stride=1, padding=kernel//2)
    mask = mask < thres  # 背景掩码（True表示背景区域）

    # 对误差绝对值进行最大池化，突出显著差异区域
    pool = torch.nn.functional.max_pool2d(
            -diff.abs(), kernel_size=kernel, stride=1, padding=kernel//2)
    pool = -pool  # 取负再取负，等效于最大池化误差绝对值
    #pool = torch.nn.functional.max_pool2d(
    #        diff.abs(), kernel_size=kernel, stride=1, padding=kernel//2)

    # 背景区域评分：取第90百分位数
    bg_mask = mask
    attn1 = (pool*bg_mask).flatten(start_dim=1).cpu()  # 背景区域误差
    attn1 = np.percentile(attn1, 90, axis=1)  # 每个样本取第90百分位数

    # 信号区域评分：取第99百分位数
    sig_mask = ~mask  # 信号掩码（非背景区域）
    attn2 = (pool*sig_mask).flatten(start_dim=1).cpu()  # 信号区域误差
    attn2 = np.percentile(attn2, 99, axis=1)  # 每个样本取第99百分位数

    # 综合评分：2*背景评分 + 信号评分
    attn = 2*attn1 + attn2
    attn_auc = roc_auc_score(label, -attn)  # 使用负注意力评分

    return mae_auc, mse_auc, xujing_auc, attn_auc


def setup_seed(seed):
    """
    设置随机种子以确保实验结果可复现
    参数:
        seed: 随机种子
    """
    torch.manual_seed(seed)           # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed(seed)      # 设置PyTorch GPU随机种子
    np.random.seed(seed)              # 设置NumPy随机种子
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作确定性
