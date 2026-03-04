"""
计算多种异常检测方法的AUC
包括VAE重建误差和传统无监督异常检测算法
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
from sklearn.metrics import roc_auc_score  # ROC-AUC计算
from sklearn.neighbors import LocalOutlierFactor  # LOF异常检测
from sklearn.svm import OneClassSVM  # 单类SVM
from sklearn.covariance import EllipticEnvelope  # 椭圆包络
from sklearn.ensemble import IsolationForest  # 孤立森林
import pickle
import os

# 设置随机种子，确保实验可复现
setup_seed(99)

# 创建结果目录
os.makedirs('./result', exist_ok=True)

# 调试断点（可注释掉）
import pdb
pdb.set_trace()

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
        shuffle=True      # 打乱数据
)

# 设置模型并加载预训练权重
vae_model = VAE().cuda()  # 创建卷积VAE模型并移至GPU
vae_model.load_state_dict(torch.load("./result/mae_best.pth"))  # 加载训练好的模型权重

# -------------------- 特征学习：使用VAE提取潜在特征 --------------------
with torch.no_grad():  # 不需要计算梯度

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

    # 保存正常样本和异常样本的图像
    save_image(data[label==0][-64:], "./result/input-test-n.png")    # 正常样本输入
    save_image(output[label==0][-64:], "./result/output-test-n.png") # 正常样本重建
    save_image(data[label==1][-64:], "./result/input-test-a.png")    # 异常样本输入
    save_image(output[label==1][-64:], "./result/output-test-a.png") # 异常样本重建

    # 计算VAE重建误差的AUC
    loss = vae_loss(output, mean, logvar, data)  # VAE损失
    mae_auc, mse_auc, xujing_auc, attn_auc = auc(data, label, output)  # 四种AUC评分

    # 打印VAE的AUC结果
    print("[Val] loss:{}, auc:{}|{}|{}|{}".format(
        loss, mae_auc, mse_auc, xujing_auc, attn_auc
    ))

# -------------------- 传统异常检测器在潜在空间上的评估 --------------------
# 特征提取：将VAE的均值和对数方差拼接作为特征
X = torch.cat([mean, logvar], dim=1).cpu().numpy()  # [batch, 150] (75+75)
label = label.cpu().numpy()  # 转换为numpy数组

# 1. LOF（局部离群因子）检测器
lof = LocalOutlierFactor(n_neighbors=500)  # 使用500个近邻
_ = lof.fit_predict(X)  # 拟合模型并预测
lof_score = lof.negative_outlier_factor_  # 离群因子（值越小越异常）

# 2. 单类SVM检测器
#svm = OneClassSVM(gamma='auto').fit(X)  # 自动选择gamma（已注释）
svm = OneClassSVM(gamma='scale').fit(X)  # 使用'scale'模式选择gamma
_ = svm.fit_predict(X)  # 拟合模型
svm_score = svm.score_samples(X)  # 样本评分（值越小越异常）

# 3. 椭圆包络（EllipticEnvelope）检测器（假设数据服从高斯分布）
cov = EllipticEnvelope(random_state=0).fit(X)  # 拟合椭圆包络模型
cov_score = cov.decision_function(X)  # 决策函数值（值越小越异常）

# 4. 孤立森林（IsolationForest）检测器
iso = IsolationForest(random_state=0).fit(X)  # 拟合孤立森林模型
iso_score = iso.score_samples(X)  # 异常评分（值越小越异常）

# 计算各种检测器的AUC（使用负分数，因为评分值越小表示越异常）
lof_auc = roc_auc_score(label, -lof_score)  # LOF AUC
svm_auc = roc_auc_score(label, -svm_score)  # 单类SVM AUC
cov_auc = roc_auc_score(label, -cov_score)  # 椭圆包络 AUC
iso_auc = roc_auc_score(label, -iso_score)  # 孤立森林 AUC

# 打印传统检测器的AUC结果
print("传统检测器AUC:")
print("LOF: {}, SVM: {}, EllipticEnvelope: {}, IsolationForest: {}".format(
    lof_auc, svm_auc, cov_auc, iso_auc
))
