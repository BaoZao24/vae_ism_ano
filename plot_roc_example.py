"""
ROC曲线绘制示例代码
展示如何在VAE项目中添加ROC曲线可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 设置中文字体（如果需要）
# 优先使用系统中可用的中文字体，最后用英文字体回退
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei',      # 文泉驿微米黑
    'Noto Sans CJK SC',         # Google Noto Sans简体中文
    'Microsoft YaHei',          # 微软雅黑（如果已安装）
    'SimHei',                   # 黑体
    'DejaVu Sans',              # 英文字体回退
    'Arial'                     # 英文字体回退
]
plt.rcParams['axes.unicode_minus'] = False

def plot_single_roc_curve(labels, scores, method_name, color='blue', ax=None):
    """
    绘制单条ROC曲线

    参数:
        labels: 真实标签 (1=正常, 0=异常)
        scores: 异常评分 (值越大越异常)
        method_name: 方法名称
        color: 曲线颜色
        ax: matplotlib轴对象，如果为None则创建新图
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = roc_auc_score(labels, scores)

    # 绘制ROC曲线
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{method_name} (AUC = {auc_score:.4f})')

    # 标记几个关键阈值点
    if len(thresholds) > 10:
        # 等间隔选择一些阈值点
        step = len(thresholds) // 10
        indices = list(range(0, len(thresholds), step))
        for idx in indices[:5]:  # 只标记前5个点
            ax.scatter(fpr[idx], tpr[idx], color=color, s=50, alpha=0.7)
            # 可以添加阈值标注
            # ax.annotate(f'{thresholds[idx]:.2f}',
            #            (fpr[idx], tpr[idx]),
            #            fontsize=8)

    return fpr, tpr, thresholds, auc_score

def plot_multiple_roc_curves(labels, scores_dict, title="ROC曲线对比",
                            save_path=None):
    """
    绘制多条ROC曲线对比图

    参数:
        labels: 真实标签 (1=正常, 0=异常)
        scores_dict: 字典 {方法名称: 异常评分数组}
        title: 图表标题
        save_path: 保存路径，如果为None则不保存
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 颜色循环
    colors = plt.cm.tab10(np.linspace(0, 1, len(scores_dict)))

    results = {}
    for (method_name, scores), color in zip(scores_dict.items(), colors):
        fpr, tpr, thresholds, auc_score = plot_single_roc_curve(
            labels, scores, method_name, color, ax)

        results[method_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc_score
        }

    # 绘制对角线（随机猜测）
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1,
            label='随机猜测 (AUC = 0.5)')

    # 设置图形属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正例率 (False Positive Rate)', fontsize=12)
    ax.set_ylabel('真正例率 (True Positive Rate)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加AUC汇总表
    auc_text = "AUC汇总:\n"
    for method_name, result in results.items():
        auc_text += f"{method_name}: {result['auc']:.4f}\n"

    ax.text(0.6, 0.15, auc_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线图已保存到: {save_path}")

    plt.show()
    return results

def plot_roc_with_threshold_analysis(labels, scores, method_name,
                                    save_path=None):
    """
    绘制ROC曲线并添加阈值分析

    参数:
        labels: 真实标签
        scores: 异常评分
        method_name: 方法名称
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 主ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = roc_auc_score(labels, scores)

    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2,
                   label=f'{method_name} (AUC = {auc_score:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('FPR')
    axes[0, 0].set_ylabel('TPR')
    axes[0, 0].set_title(f'{method_name} ROC曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 阈值-TPR/FPR关系
    axes[0, 1].plot(thresholds, tpr, 'g-', label='TPR', linewidth=2)
    axes[0, 1].plot(thresholds, fpr, 'r-', label='FPR', linewidth=2)
    axes[0, 1].set_xlabel('阈值')
    axes[0, 1].set_ylabel('率')
    axes[0, 1].set_title('阈值 vs TPR/FPR')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 阈值分布
    axes[1, 0].hist(scores[labels==0], bins=30, alpha=0.7,
                   label='正常样本', color='green')
    axes[1, 0].hist(scores[labels==1], bins=30, alpha=0.7,
                   label='异常样本', color='red')
    axes[1, 0].set_xlabel('异常评分')
    axes[1, 0].set_ylabel('样本数')
    axes[1, 0].set_title('评分分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 最佳阈值分析（最大化Youden指数）
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]

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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"详细ROC分析图已保存到: {save_path}")

    plt.show()

    return {
        'best_threshold': best_threshold,
        'best_tpr': tpr[best_idx],
        'best_fpr': fpr[best_idx],
        'youden_index': youden_index[best_idx]
    }

# ==================== 示例使用 ====================

if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000

    # 正常样本评分（值较小）
    normal_scores = np.random.normal(0.3, 0.1, n_samples//2)
    # 异常样本评分（值较大）
    anomaly_scores = np.random.normal(0.7, 0.2, n_samples//2)

    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([np.zeros(n_samples//2),  # 0=正常
                                 np.ones(n_samples//2)])  # 1=异常

    # 模拟多种检测方法
    scores_dict = {
        '方法A': all_scores,
        '方法B': all_scores + np.random.normal(0, 0.05, n_samples),
        '方法C': all_scores + np.random.normal(0, 0.1, n_samples),
        '方法D': all_scores + np.random.normal(0.1, 0.15, n_samples)
    }

    print("示例1: 多条ROC曲线对比")
    print("-" * 50)
    results = plot_multiple_roc_curves(
        all_labels, scores_dict,
        title="异常检测方法ROC曲线对比",
        save_path="./roc_comparison_example.png"
    )

    print("\n示例2: 单方法详细分析")
    print("-" * 50)
    best_threshold_info = plot_roc_with_threshold_analysis(
        all_labels, scores_dict['方法A'], '方法A',
        save_path="./roc_detailed_analysis.png"
    )

    print(f"\n最佳阈值信息:")
    print(f"  阈值: {best_threshold_info['best_threshold']:.4f}")
    print(f"  TPR: {best_threshold_info['best_tpr']:.4f}")
    print(f"  FPR: {best_threshold_info['best_fpr']:.4f}")
    print(f"  Youden指数: {best_threshold_info['youden_index']:.4f}")

    # 应用最佳阈值
    predictions = (scores_dict['方法A'] >= best_threshold_info['best_threshold']).astype(int)
    accuracy = np.mean(predictions == all_labels)
    print(f"\n应用最佳阈值后的准确率: {accuracy:.4f}")