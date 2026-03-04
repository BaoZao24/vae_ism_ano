"""
VAE模型训练脚本
用于无监督频谱异常检测
"""

import torch
from torchvision.utils import save_image
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from model import VAE, VAE_FC, vae_loss
from util import auc, setup_seed, forward
from tqdm import tqdm
import torch.nn as nn
import random
import pdb
import pickle
import os

# 设置随机种子，确保实验可复现
setup_seed(99)

# 创建结果目录
os.makedirs('./result', exist_ok=True)

# 加载数据集（16QAM调制信号的频谱数据）
with open("../data/16QAM_Train_Test.pkl", "rb") as f:
    data = pickle.load(f)

# 提取训练集和测试集
train_data = data["train_data"]    # 训练数据
train_label = data["train_label"]  # 训练标签 (0=正常, 1=异常)
test_data = data["test_data"]      # 测试数据
test_label = data["test_label"]    # 测试标签

# 设置数据加载器
train_loader = torch.utils.data.DataLoader(
        dataset=TensorDataset(train_data, train_label),  # 训练数据集
        batch_size=64,    # 批大小
        shuffle=True      # 打乱数据
)
test_loader = torch.utils.data.DataLoader(
        dataset=TensorDataset(test_data, test_label),    # 测试数据集
        batch_size=64,    # 批大小
        shuffle=True      # 打乱数据（评估时也需要shuffle）
)

# 设置模型
vae_model = VAE().cuda()  # 创建卷积VAE模型并移至GPU
#vae_model.load_state_dict(torch.load("png-flip-pow-seperate-shufflenobn/attn_best.pth"))  # 加载预训练模型（已注释）

# 设置TensorBoard记录器
writer = SummaryWriter("./result")

# 设置优化器
# TODO use l2 reg（可添加L2正则化）
optimizer = Adam(vae_model.parameters(), lr=1e-4)  # Adam优化器，学习率1e-4

# 记录最佳MAE AUC
mae_best = 0

# 开始训练（共1000个epoch）
for epoch in range(1000):

    # -------------------- 训练阶段 --------------------
    for idx, (data, label) in enumerate(train_loader):

        # 前向传播
        (output, mean, logvar), data = forward(vae_model, data)  # 获取重建图像和潜在变量
        loss = vae_loss(output, mean, logvar, data)              # 计算VAE损失

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 更新参数

        # 打印训练进度
        print("Epoch:{} [{}/{}], loss:{}".format(
            epoch, idx, len(train_loader), loss
        ))

        # 记录训练损失到TensorBoard
        step = idx + len(train_loader) * epoch  # 计算全局步数
        writer.add_scalars("loss", {"train": loss.item()}, step)

    # 每个epoch保存一次训练样本的输入和重建图像
    save_image(data, "./result/input-train.png")    # 保存输入图像
    save_image(output, "./result/output-train.png") # 保存重建图像

    # -------------------- 验证阶段（每5个epoch一次） --------------------
    if epoch % 5 == 4:  # epoch索引从0开始，所以4表示第5个epoch

        with torch.no_grad():  # 验证阶段不需要计算梯度

            # 收集所有测试批次的数据
            summary = []
            for data, label in test_loader:
                (output, mean, logvar), data = forward(vae_model, data)  # 前向传播
                summary.append([data, label, output, mean, logvar])      # 保存批次结果

            # 合并所有批次的数据
            summary = zip(*summary)  # 转置：将列表的列表转换为按字段分组
            summary = map(
                    lambda x: torch.cat(x, dim=0),  # 沿批次维度拼接
                    list(summary)
            )
            data, label, output, mean, logvar = list(summary)  # 解包得到完整测试集
            
            # 保存正常样本（label=0）和异常样本（label=1）的图像
            save_image(data[label==0][-64:], "./result/input-test-n.png")    # 正常样本输入
            save_image(output[label==0][-64:], "./result/output-test-n.png") # 正常样本重建
            save_image(data[label==1][-64:], "./result/input-test-a.png")    # 异常样本输入
            save_image(output[label==1][-64:], "./result/output-test-a.png") # 异常样本重建

            # 计算验证损失和多种AUC评分
            loss = vae_loss(output, mean, logvar, data)  # 验证集损失
            mae_auc, mse_auc, xujing_auc, attn_auc = auc(data, label, output)  # 计算四种AUC

            # 打印验证结果
            print("[Val] Epoch:{}, loss:{}, auc:{}|{}|{}|{}".format(
                epoch, loss, mae_auc, mse_auc, xujing_auc, attn_auc
            ))

            # 记录验证指标到TensorBoard
            writer.add_scalars("loss", {"test": loss.item()}, step)  # 验证损失
            writer.add_scalars("auc", {"mae": mae_auc.item()}, step)      # MAE AUC
            writer.add_scalars("auc", {"attn": attn_auc.item()}, step)    # 注意力AUC
            writer.add_scalars("auc", {"xujing": xujing_auc.item()}, step) # 相对误差AUC
            writer.add_scalars("auc", {"mse": mse_auc.item()}, step)      # MSE AUC

        # 如果当前MAE AUC优于历史最佳，则保存模型
        if mae_auc > mae_best:
            mae_best = mae_auc  # 更新最佳AUC
            torch.save(vae_model.state_dict(), "./result/mae_best.pth")  # 保存模型权重 
