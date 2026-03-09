"""
VAE重建误差ROC曲线绘制脚本
只计算和绘制四种VAE重建误差的ROC曲线，移除传统检测器方法
"""

import torch
from torchvision.utils import save_image
from torch.utils.data import TensorDataset
from model import VAE
from util import forward, setup_seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import pickle
import os
import json

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei',      # 文泉驿微米黑
    'Noto Sans CJK SC',         # Google Noto Sans简体中文
    'Microsoft YaHei',          # 微软雅黑（如果已安装）
    'SimHei',                   # 黑体
    'DejaVu Sans',              # 英文字体回退
    'Arial'                     # 英文字体回退
]
plt.rcParams['axes.unicode_minus'] = False

def plot_vae_roc_curves(labels, scores_dict, title="VAE重建误差ROC曲线对比",
                        save_path="./result/vae_reconstruction_roc_curves.png"):
    """
    绘制VAE重建误差的ROC曲线对比图

    参数:
        labels: 真实标签 (1=正常, 0=异常)
        scores_dict: 字典 {方法名称: 异常评分数组}，评分值越大表示越异常
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))

    # 颜色循环，为四种VAE误差分配不同颜色
    colors = ['blue', 'green', 'orange', 'red']

    results = {}
    for (method_name, scores), color in zip(scores_dict.items(), colors):
        # 计算ROC曲线（评分值越大越异常，将异常样本label=0视为正例）
        # 因此使用1-labels将异常转换为正例
        fpr, tpr, thresholds = roc_curve(1 - labels, scores)
        auc_score = roc_auc_score(1 - labels, scores)

        # 绘制曲线
        plt.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{method_name} (AUC = {auc_score:.4f})')

        # 保存结果
        results[method_name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(auc_score)
        }

    # 绘制对角线（随机猜测）
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1,
            label='随机猜测 (AUC = 0.5)')

    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # 添加AUC汇总表
    auc_text = "AUC汇总:\n"
    for method_name, result in results.items():
        auc_text += f"{method_name}: {result['auc']:.4f}\n"

    plt.text(0.6, 0.15, auc_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            transform=plt.gca().transAxes)

    plt.tight_layout()

    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"VAE ROC曲线图已保存到: {save_path}")

    # 保存详细数据
    data_path = save_path.replace('.png', '_data.json')
    with open(data_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"VAE ROC曲线数据已保存到: {data_path}")

    return results

def compute_vae_scores(x, y):
    """
    计算VAE的四种重建误差评分
    注意：这些评分值越大表示越异常

    参数:
        x: 原始图像 [batch, 1, H, W]
        y: 重建图像 [batch, 1, H, W]
    返回:
        mae_scores: MAE评分（误差越大越异常）
        mse_scores: MSE评分
        xujing_scores: 相对误差评分
        attn_scores: 注意力评分
    """
    diff = x - y  # 重建误差

    # 1. MAE（平均绝对误差）评分
    mae_scores = diff.abs().sum(dim=[1,2,3]).detach().cpu().numpy()

    # 2. MSE（均方误差）评分
    mse_scores = diff.abs().pow(2).sum(dim=[1,2,3]).detach().cpu().numpy()

    # 3. 相对误差评分（xujing）
    # 避免除以零
    x_safe = x.clone()
    x_safe[x_safe == 0] = 1e-8
    xujing_scores = (diff / x_safe).abs().sum(dim=[1,2,3]).detach().cpu().numpy()

    # 4. 注意力评分（结合空间信息）
    thres = 0.05
    kernel = 3
    mask = torch.nn.functional.max_pool2d(
            x, kernel_size=kernel, stride=1, padding=kernel//2)
    mask = mask < thres

    pool = torch.nn.functional.max_pool2d(
            -diff.abs(), kernel_size=kernel, stride=1, padding=kernel//2)
    pool = -pool

    # 背景区域评分：第90百分位数
    bg_mask = mask
    attn1 = (pool*bg_mask).flatten(start_dim=1).cpu().numpy()
    attn1 = np.percentile(attn1, 90, axis=1)

    # 信号区域评分：第99百分位数
    sig_mask = ~mask
    attn2 = (pool*sig_mask).flatten(start_dim=1).cpu().numpy()
    attn2 = np.percentile(attn2, 99, axis=1)

    # 综合评分：2*背景评分 + 信号评分
    attn_scores = 2*attn1 + attn2

    return mae_scores, mse_scores, xujing_scores, attn_scores

def plot_detailed_vae_analysis(labels, mae_scores, mse_scores, xujing_scores, attn_scores,
                              save_dir="./result/vae_detailed_analysis"):
    """
    为四种VAE重建误差绘制详细分析图

    参数:
        labels: 真实标签 (1=正常, 0=异常)
        mae_scores: MAE评分
        mse_scores: MSE评分
        xujing_scores: 相对误差评分
        attn_scores: 注意力评分
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    scores_dict = {
        'VAE-MAE': mae_scores,
        'VAE-MSE': mse_scores,
        'VAE-xujing': xujing_scores,
        'VAE-attn': attn_scores
    }

    detailed_results = {}

    for method_name, scores in scores_dict.items():
        print(f"  分析 {method_name}...")

        # 计算ROC曲线（评分值越大越异常，将异常样本label=0视为正例）
        fpr, tpr, thresholds = roc_curve(1 - labels, scores)
        auc_score = roc_auc_score(1 - labels, scores)

        # 计算Youden指数（最佳阈值）
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 主ROC曲线
        axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2,
                       label=f'{method_name} (AUC = {auc_score:.4f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机猜测')
        axes[0, 0].set_xlabel('FPR')
        axes[0, 0].set_ylabel('TPR')
        axes[0, 0].set_title(f'{method_name} ROC曲线')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)

        # 标记最佳阈值点
        axes[0, 0].scatter(fpr[best_idx], tpr[best_idx], color='red', s=100,
                          zorder=5, label=f'最佳阈值: {best_threshold:.3f}')

        # 2. 阈值-TPR/FPR关系
        axes[0, 1].plot(thresholds, tpr, 'g-', label='TPR', linewidth=2)
        axes[0, 1].plot(thresholds, fpr, 'r-', label='FPR', linewidth=2)
        axes[0, 1].axvline(x=best_threshold, color='red', linestyle='--',
                          label=f'最佳阈值: {best_threshold:.3f}')
        axes[0, 1].set_xlabel('阈值')
        axes[0, 1].set_ylabel('率')
        axes[0, 1].set_title('阈值 vs TPR/FPR')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 评分分布
        axes[1, 0].hist(scores[labels==1], bins=30, alpha=0.7,
                       label='正常样本', color='green')
        axes[1, 0].hist(scores[labels==0], bins=30, alpha=0.7,
                       label='异常样本', color='red')
        axes[1, 0].axvline(x=best_threshold, color='red', linestyle='--',
                          label=f'最佳阈值: {best_threshold:.3f}')
        axes[1, 0].set_xlabel('异常评分')
        axes[1, 0].set_ylabel('样本数')
        axes[1, 0].set_title('评分分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Youden指数分析
        axes[1, 1].plot(thresholds, youden_index, 'purple', linewidth=2)
        axes[1, 1].axvline(x=best_threshold, color='red', linestyle='--',
                          label=f'最佳阈值: {best_threshold:.3f}')
        axes[1, 1].set_xlabel('阈值')
        axes[1, 1].set_ylabel('Youden指数 (TPR-FPR)')
        axes[1, 1].set_title('最佳阈值分析')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{method_name} - 详细ROC分析', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(save_dir, f"{method_name.lower().replace('-', '_')}_detailed_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    详细分析图已保存到: {save_path}")

        detailed_results[method_name] = {
            'best_threshold': float(best_threshold),
            'best_fpr': float(fpr[best_idx]),
            'best_tpr': float(tpr[best_idx]),
            'youden_index': float(youden_index[best_idx]),
            'auc': float(auc_score)
        }

    return detailed_results

# 主程序
def main():
    # 设置随机种子，确保实验可复现
    setup_seed(99)

    # 创建结果目录
    os.makedirs('./result', exist_ok=True)

    # 加载数据集（16QAM调制信号的频谱数据）
    print("加载数据集...")
    with open("../data/16QAM_Train_Test.pkl", "rb") as f:
        data = pickle.load(f)

    # 提取训练集和测试集
    train_data = data["train_data"]    # 训练数据
    train_label = data["train_label"]  # 训练标签 (0=异常, 1=正常)
    test_data = data["test_data"]      # 测试数据
    test_label = data["test_label"]    # 测试标签

    # 设置数据加载器
    test_loader = torch.utils.data.DataLoader(
            dataset=TensorDataset(test_data, test_label),    # 测试数据集
            batch_size=64,    # 批大小
            shuffle=False     # 不打乱，保持顺序
    )

    # 设置模型并加载预训练权重
    print("加载VAE模型...")
    vae_model = VAE().cuda()
    vae_model.load_state_dict(torch.load("./result/mae_best.pth"))

    # -------------------- 提取测试集特征和重建 --------------------
    print("提取测试集特征和重建...")
    with torch.no_grad():
        # 收集所有测试批次的数据
        summary = []
        for data_batch, label_batch in test_loader:
            (output, mean, logvar), data_batch = forward(vae_model, data_batch)
            summary.append([data_batch, label_batch, output, mean, logvar])

        # 合并所有批次的数据
        summary = zip(*summary)
        summary = map(
                lambda x: torch.cat(x, dim=0),
                list(summary)
        )
        test_data_tensor, test_label_tensor, output_tensor, mean_tensor, logvar_tensor = list(summary)

    # 保存异常样本和正常样本的图像（可选）
    print("保存重建图像...")
    save_image(test_data_tensor[test_label_tensor==0][-64:], "./result/input-test-anomaly.png")
    save_image(output_tensor[test_label_tensor==0][-64:], "./result/output-test-anomaly.png")
    save_image(test_data_tensor[test_label_tensor==1][-64:], "./result/input-test-normal.png")
    save_image(output_tensor[test_label_tensor==1][-64:], "./result/output-test-normal.png")

    # 转换为numpy数组
    test_labels = test_label_tensor.cpu().numpy()

    # -------------------- 计算VAE重建误差评分 --------------------
    print("\n=== 计算VAE重建误差评分 ===")
    mae_scores, mse_scores, xujing_scores, attn_scores = compute_vae_scores(
        test_data_tensor, output_tensor)

    # 调试信息：查看标签分布和平均误差
    print(f"标签分布: 正常样本(label=1): {(test_labels == 1).sum()}, 异常样本(label=0): {(test_labels == 0).sum()}")
    print(f"正常样本MAE平均值: {mae_scores[test_labels == 1].mean():.4f}")
    print(f"异常样本MAE平均值: {mae_scores[test_labels == 0].mean():.4f}")
    print(f"正常样本MSE平均值: {mse_scores[test_labels == 1].mean():.4f}")
    print(f"异常样本MSE平均值: {mse_scores[test_labels == 0].mean():.4f}")

    print(f"MAE评分范围: [{mae_scores.min():.4f}, {mae_scores.max():.4f}]")
    print(f"MSE评分范围: [{mse_scores.min():.4f}, {mse_scores.max():.4f}]")
    print(f"xujing评分范围: [{xujing_scores.min():.4f}, {xujing_scores.max():.4f}]")
    print(f"attn评分范围: [{attn_scores.min():.4f}, {attn_scores.max():.4f}]")

    # 计算AUC（使用负评分，因为重建误差值越大越异常，而AUC需要评分值越大越正常）
    # 注意：这里计算AUC只是为了与原始cal_auc.py结果对比
    # 实际绘图时会使用正确的评分方向
    mae_auc = roc_auc_score(test_labels, -mae_scores)
    mse_auc = roc_auc_score(test_labels, -mse_scores)
    xujing_auc = roc_auc_score(test_labels, -xujing_scores)
    attn_auc = roc_auc_score(test_labels, -attn_scores)

    print(f"\nVAE重建误差AUC (使用负评分，与util.py一致):")
    print(f"  MAE AUC:  {mae_auc:.4f}")
    print(f"  MSE AUC:  {mse_auc:.4f}")
    print(f"  xujing AUC: {xujing_auc:.4f}")
    print(f"  attn AUC:   {attn_auc:.4f}")

    # -------------------- ROC曲线绘制 --------------------
    print("\n=== 绘制VAE重建误差ROC曲线 ===")

    # 准备评分字典
    # 注意：所有评分需要统一为值越大越异常
    scores_dict = {
        'VAE-MAE': mae_scores,
        'VAE-MSE': mse_scores,
        'VAE-xujing': xujing_scores,
        'VAE-attn': attn_scores
    }

    # 标签：1=正常，0=异常
    labels = test_labels

    # 绘制综合ROC曲线
    print("绘制综合ROC曲线对比图...")
    roc_results = plot_vae_roc_curves(
        labels=labels,
        scores_dict=scores_dict,
        title="VAE重建误差ROC曲线对比",
        save_path="./result/vae_reconstruction_roc_curves.png"
    )

    # 为每个VAE误差方法绘制详细分析图
    print("\n绘制各VAE误差方法详细分析图...")
    detailed_results = plot_detailed_vae_analysis(
        labels=labels,
        mae_scores=mae_scores,
        mse_scores=mse_scores,
        xujing_scores=xujing_scores,
        attn_scores=attn_scores,
        save_dir="./result/vae_detailed_analysis"
    )

    # -------------------- 保存结果 --------------------
    print("\n=== 保存完整结果 ===")

    # 保存AUC结果
    results = {
        "model_type": "VAE",
        "test_samples": len(test_labels),
        "test_normal": int((test_labels == 1).sum()),
        "test_anomaly": int((test_labels == 0).sum()),
        "vae_auc_scores": {
            "mae": float(mae_auc),
            "mse": float(mse_auc),
            "xujing": float(xujing_auc),
            "attn": float(attn_auc)
        },
        "roc_curves": roc_results,
        "detailed_analysis": detailed_results
    }

    with open("./result/vae_reconstruction_roc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"完整结果已保存到: ./result/vae_reconstruction_roc_results.json")

    # -------------------- 最佳阈值应用示例 --------------------
    print("\n=== 最佳阈值应用示例（以VAE-attn为例，性能最佳） ===")

    # 以VAE-attn为例，展示如何使用最佳阈值
    attn_scores = scores_dict['VAE-attn']
    attn_best_threshold = detailed_results['VAE-attn']['best_threshold']

    # 使用最佳阈值进行预测（评分值越大越异常）
    predictions = (attn_scores >= attn_best_threshold).astype(int)

    print(f"VAE-attn最佳阈值: {attn_best_threshold:.4f}")
    print(f"VAE-attn AUC: {detailed_results['VAE-attn']['auc']:.4f}")

    # 注意：预测结果中，1表示异常（因为评分值≥阈值），0表示正常
    # 而原始标签中，1=正常，0=异常，所以需要调整
    # 对于混淆矩阵，我们需要将预测结果映射为与标签相同的定义
    # 即：将预测的异常(1)映射为标签的异常(0)，预测的正常(0)映射为标签的正常(1)
    predictions_adjusted = 1 - predictions  # 反转预测结果

    print("混淆矩阵:")
    cm = confusion_matrix(labels, predictions_adjusted)
    print(cm)

    print("\n分类报告:")
    print(classification_report(labels, predictions_adjusted,
                              target_names=['异常', '正常'],
                              digits=4))

    # 计算各指标
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    print("\n=== 完成 ===")
    print("VAE重建误差ROC曲线和相关分析图已保存到 ./result/ 目录")
    print("运行完成！")

if __name__ == "__main__":
    main()