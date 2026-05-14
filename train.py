"""
中国交通标志检测训练脚本
基于 YOLO26n + PyTorch
优化适配 RTX 4060 (8GB VRAM)
"""
import os
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from ultralytics import YOLO

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_fonts():
    """中文字体（Windows/Linux 兼容）"""
    font_candidates = [
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for fpath in font_candidates:
        if Path(fpath).exists():
            try:
                fm.fontManager.addfont(fpath)
                plt.rcParams['font.sans-serif'] = [
                    fm.FontProperties(fname=fpath).get_name(),
                    'Microsoft YaHei',
                    'SimHei',
                    'DejaVu Sans',
                ]
                plt.rcParams['axes.unicode_minus'] = False
                return
            except Exception:
                continue
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


setup_fonts()


def pick_device():
    return 0 if torch.cuda.is_available() else 'cpu'


def check_gpu():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu_name}")
        print(f"[显存] {gpu_memory:.1f} GB")
        return True
    else:
        print("[警告] 未检测到GPU，将使用CPU训练")
        return False


def plot_training_curves(results_csv):
    """绘制训练曲线图：损失 + mAP50 + 精确率与召回率"""
    if not results_csv.exists():
        print("[警告] 未找到训练结果文件 results.csv，跳过绘图")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    save_dir = results_csv.parent / 'training_curves.png'

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 左图：训练/验证损失
    ax = axes[0]
    if 'train/box_loss' in df.columns:
        ax.plot(df['epoch'], df['train/box_loss'], label='训练边界框损失', linewidth=1.5, color='#2196f3')
    if 'val/box_loss' in df.columns:
        ax.plot(df['epoch'], df['val/box_loss'], label='验证边界框损失', linewidth=1.5, color='#ff9800', linestyle='--')
    if 'train/cls_loss' in df.columns:
        ax.plot(df['epoch'], df['train/cls_loss'], label='训练分类损失', linewidth=1.5, color='#4caf50', alpha=0.6)
    if 'val/cls_loss' in df.columns:
        ax.plot(df['epoch'], df['val/cls_loss'], label='验证分类损失', linewidth=1.5, color='#f44336', linestyle='--', alpha=0.6)
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('损失值')
    ax.set_title('损失曲线')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 中图：mAP
    ax = axes[1]
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2, color='#2196f3')
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('平均精度')
    ax.set_title('mAP50 曲线')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 右图：精确率 + 召回率
    ax = axes[2]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/precision(B)'], label='精确率', linewidth=2, color='#4caf50')
    if 'metrics/recall(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/recall(B)'], label='召回率', linewidth=2, color='#9c27b0', linestyle='--')
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('精确率 / 召回率')
    ax.set_title('精确率与召回率曲线')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(save_dir, dpi=150)
    plt.close(fig)
    print(f"[曲线图] 已保存: {save_dir}")


def print_val_metrics(metrics):
    """打印验证指标"""
    print("\n" + "=" * 50)
    print("最佳模型验证指标")
    print("=" * 50)
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  精确率:    {metrics.box.mp:.3f}")
    if hasattr(metrics.box, 'mr'):
        print(f"  召回率:    {metrics.box.mr:.3f}")
    print()
    class_names = {'mandatory': '指示标志', 'prohibitory': '禁止标志', 'warning': '警告标志'}
    for i, name in enumerate(['mandatory', 'prohibitory', 'warning']):
        ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0.0
        print(f"  {class_names[name]} AP50: {ap:.3f}")


def main():
    print("=" * 50)
    print("中国交通标志检测系统 - YOLO26n 训练")
    print("=" * 50)

    check_gpu()
    device = pick_device()

    print("\n[加载] 预训练模型: yolo26n.pt（首次运行将自动下载）")
    model = YOLO('yolo26n.pt')

    print("\n[训练开始] 正在训练中...")
    results = model.train(
        data='cctsdb2021.yaml',

        epochs=50,
        patience=5,
        save_period=10,
        close_mosaic=30,

        imgsz=640,
        batch=16,
        workers=4,
        cache=False,
        amp=True,

        device=device,

        verbose=True,

        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        cos_lr=True,

        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,

        save=True,
        plots=False,

        project='.',
        name='train',
        exist_ok=True,

        pretrained=True,
        deterministic=False,
        seed=0,
    )

    print("\n[训练完成]")
    save_dir = Path(results.save_dir)
    best_path = save_dir / 'weights' / 'best.pt'
    last_path = save_dir / 'weights' / 'last.pt'
    print(f"[模型路径] 最佳: {best_path}")
    print(f"[模型路径] 最后: {last_path}")

    # 用最佳模型做验证评估
    if best_path.exists():
        print("\n[验证] 加载最佳模型进行评估...")
        best_model = YOLO(str(best_path))
        metrics = best_model.val(
            data='cctsdb2021.yaml',
            split='val',
            imgsz=640,
            batch=16,
            device=device,
            verbose=False,
            plots=False,
            save=False,
        )
        print_val_metrics(metrics)

    # 绘制训练曲线图
    plot_training_curves(save_dir / 'results.csv')


if __name__ == '__main__':
    main()
