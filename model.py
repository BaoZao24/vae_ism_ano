"""
变分自编码器（VAE）模型定义
用于无监督频谱异常检测
"""

import torch
import torch.nn as nn


def basic_conv(inc, out_c):
    """
    基础卷积块
    参数:
        inc: 输入通道数
        out_c: 输出通道数
    返回:
        Sequential: 包含Conv2d(4x4, stride=2, padding=1) + LeakyReLU的模块
    """
    block = nn.Sequential(
        nn.Conv2d(inc, out_c, 4, 2, 1),  # 4x4卷积，步长2，填充1（下采样）
        #nn.BatchNorm2d(out_c),  # 批归一化（已注释）
        nn.LeakyReLU(0.2)  # LeakyReLU激活函数
    )

    return block


def basic_deconv(inc, out_c):
    """
    基础反卷积块（转置卷积）
    参数:
        inc: 输入通道数
        out_c: 输出通道数
    返回:
        Sequential: 包含ConvTranspose2d(4x4, stride=2, padding=1) + LeakyReLU的模块
    """
    block = nn.Sequential(
        nn.ConvTranspose2d(inc, out_c, 4, 2, 1),  # 4x4转置卷积，步长2，填充1（上采样）
        #nn.BatchNorm2d(out_c),  # 批归一化（已注释）
        nn.LeakyReLU(0.2)  # LeakyReLU激活函数
    )

    return block


class VAE(nn.Module):
    """
    卷积变分自编码器（VAE）模型
    使用卷积层编码，转置卷积层解码
    用于处理二维频谱图像数据（64x64）
    """

    def __init__(self):
        """初始化VAE模型架构"""
        super().__init__()

        ch = 32          # 基础通道数
        depth = 4         # 卷积层深度
        bottle = 75       # 隐变量（潜在空间）维度
        flat = ch * (2**depth)  # 编码器输出的扁平化维度
        self.flat = flat  # 保存扁平化维度供解码器使用
        
        # 编码器：多个卷积层下采样，最后接一个4x4卷积将特征图压缩为1x1
        self.encoder = nn.Sequential(
            *[basic_conv(ch*2**(i-1) if i>1 else 1, ch*2**i)
                for i in range(1, depth+1)],  # 生成depth个卷积块，每层通道数翻倍
            nn.Conv2d(flat, flat, 4, 4, 0)    # 将特征图从4x4压缩到1x1
        )

        # 全连接层：将1x1特征图转换为4096维特征向量
        self.fc1 = nn.Sequential(
            nn.Linear(flat, 4096),  # 全连接层
            nn.LeakyReLU(0.2),      # 激活函数
        )
        
        # 均值头：将4096维特征映射到潜在空间均值
        self.to_mean = nn.Sequential(
            nn.Linear(4096, bottle),  # 输出均值向量
        )

        # 对数方差头：将4096维特征映射到潜在空间对数方差
        self.to_log = nn.Sequential(
            nn.Linear(4096, bottle),  # 输出对数方差向量
        )

        # 从潜在空间解码：将隐变量映射回解码器输入维度
        self.from_bottle = nn.Sequential(
            nn.Linear(bottle, flat),  # 从bottle维度扩展到flat维度
            nn.LeakyReLU(0.2)         # 激活函数
        )

        # 解码器：转置卷积层上采样，恢复原始图像尺寸
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(flat, flat, 4, 4, 0),  # 将1x1特征图扩展到4x4
            *[basic_deconv(ch*2**i, ch*2**(i-1))
                for i in range(depth, 0, -1)],  # depth个反卷积块，每层通道数减半
            nn.Conv2d(ch, 1, 1, 1, 0),          # 1x1卷积将通道数从ch降到1
            nn.Sigmoid()                         # Sigmoid激活，输出值在[0,1]范围
        )

    def resample(self, mean, logvar):
        """
        重参数化技巧：从正态分布N(mean, exp(logvar))中采样
        参数:
            mean: 均值向量
            logvar: 对数方差向量
        返回:
            采样得到的隐变量z
        """
        std = logvar.mul(0.5).exp()  # 计算标准差：exp(logvar/2)
        eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准正态分布采样噪声
        return eps.mul(std).add(mean)  # z = mean + std * eps
    
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入图像 [batch_size, 1, 64, 64]
        返回:
            y: 重建图像 [batch_size, 1, 64, 64]
            mean: 潜在空间均值 [batch_size, bottle]
            logvar: 潜在空间对数方差 [batch_size, bottle]
        """
        # 编码过程
        x = self.encoder(x)                     # 卷积编码
        x = torch.flatten(x, start_dim=1)       # 展平为向量 [batch_size, flat]
        x = self.fc1(x)                         # 全连接层 [batch_size, 4096]

        # 潜在空间
        mean = self.to_mean(x)                  # 计算均值 [batch_size, bottle]
        logvar = self.to_log(x)                 # 计算对数方差 [batch_size, bottle]
        z = self.resample(mean, logvar)         # 重参数化采样 [batch_size, bottle]

        # 解码过程
        y = self.from_bottle(z)                 # 从隐变量扩展 [batch_size, flat]
        y = y.reshape(-1, self.flat, 1, 1)      # 重塑为1x1特征图 [batch_size, flat, 1, 1]
        y = self.decoder(y)                     # 反卷积解码 [batch_size, 1, 64, 64]

        return y, mean, logvar


def basic_fc(inc, out_c):
    """
    基础全连接块
    参数:
        inc: 输入维度
        out_c: 输出维度
    返回:
        Sequential: 包含Linear + LeakyReLU的模块
    """
    block = nn.Sequential(
        nn.Linear(inc, out_c),  # 全连接层
        nn.LeakyReLU(0.2)       # LeakyReLU激活函数
    )

    return block


class VAE_FC(nn.Module):
    """
    全连接变分自编码器（VAE）模型
    使用全连接层编码和解码，适用于频谱数据
    """

    def __init__(self):
        """初始化VAE_FC模型架构"""
        super().__init__()

        size = 64        # 输入图像尺寸（假设为64x64）
        bottle = 75       # 隐变量（潜在空间）维度
        factor = 2        # 全连接层宽度因子
        self.size = size  # 保存尺寸供reshape使用
        
        # 编码器：多个全连接层逐步降维
        self.encoder = nn.Sequential(
            basic_fc(size**2, 1024*factor),      # 4096 -> 2048 (64x64=4096)
            basic_fc(1024*factor, 512*factor),   # 2048 -> 1024
            basic_fc(512*factor, 256*factor),    # 1024 -> 512
            basic_fc(256*factor, 128*factor),    # 512 -> 256
        )

        
        # 均值头：将256维特征映射到潜在空间均值
        self.to_mean = nn.Sequential(
            nn.Linear(128*factor, bottle),  # 256 -> 75
        )

        # 对数方差头：将256维特征映射到潜在空间对数方差
        self.to_log = nn.Sequential(
            nn.Linear(128*factor, bottle),  # 256 -> 75
        )

        # 从潜在空间解码：将隐变量映射回解码器输入维度
        self.from_bottle = nn.Sequential(
            nn.Linear(bottle, 128*factor),  # 75 -> 256
        )

        # 解码器：多个全连接层逐步升维，恢复原始维度
        self.decoder = nn.Sequential(
            basic_fc(128*factor, 256*factor),    # 256 -> 512
            basic_fc(256*factor, 512*factor),    # 512 -> 1024
            basic_fc(512*factor, 1024*factor),   # 1024 -> 2048
            basic_fc(1024*factor, size**2)       # 2048 -> 4096 (64x64)
        )

        
    def resample(self, mean, logvar):
        """
        重参数化技巧：从正态分布N(mean, exp(logvar))中采样
        参数:
            mean: 均值向量
            logvar: 对数方差向量
        返回:
            采样得到的隐变量z
        """
        std = logvar.mul(0.5).exp()  # 计算标准差：exp(logvar/2)
        eps = torch.cuda.FloatTensor(std.size()).normal_()  # 从标准正态分布采样噪声
        return eps.mul(std).add(mean)  # z = mean + std * eps
    
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入图像 [batch_size, 1, 64, 64]
        返回:
            y: 重建图像 [batch_size, 1, 64, 64]
            mean: 潜在空间均值 [batch_size, bottle]
            logvar: 潜在空间对数方差 [batch_size, bottle]
        """
        # 编码过程
        x = torch.flatten(x, start_dim=1)       # 展平为向量 [batch_size, 4096] (64x64)
        x = self.encoder(x)                     # 全连接编码 [batch_size, 256]

        # 潜在空间
        mean = self.to_mean(x)                  # 计算均值 [batch_size, 75]
        logvar = self.to_log(x)                 # 计算对数方差 [batch_size, 75]
        z = self.resample(mean, logvar)         # 重参数化采样 [batch_size, 75]

        # 解码过程
        y = self.from_bottle(z)                 # 从隐变量扩展 [batch_size, 256]
        y = self.decoder(y)                     # 全连接解码 [batch_size, 4096]
        y = y.reshape(-1, 1, self.size, self.size)  # 重塑为图像 [batch_size, 1, 64, 64]

        return y, mean, logvar


def vae_loss(y, mean, logvar, x):
    """
    VAE损失函数：重构损失 + KL散度
    参数:
        y: 重建图像
        mean: 潜在空间均值
        logvar: 潜在空间对数方差
        x: 原始图像
    返回:
        总损失 = MSE重构损失 - λ * KL散度
    """
    # TODO tune this hypermeter
    lmd = 0.1  # KL散度权重系数（需要调优）
    mse = ((y-x)**2).sum()  # 均方误差重构损失
    kld = 0.5*(1+logvar-mean**2-logvar.exp()).sum()  # KL散度（相对标准正态分布）
    return mse - lmd*kld  # 总损失（负号因为要最大化ELBO）

