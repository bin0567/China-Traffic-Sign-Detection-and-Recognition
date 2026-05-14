"""
测试集定量评估 + 可视化拼接图生成
"""
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

BEST_PT = Path('runs/detect/train/weights/best.pt')
TEST_DATA_YAML = 'cctsdb2021_test.yaml'
VAL_BATCH = 8
VAL_IMGSZ = 640
VIS_RUN_NAME = 'test_vis'


def pick_device():
    return 0 if torch.cuda.is_available() else 'cpu'


def _save_montage(plotted_list, save_dir, idx):
    """将检测结果图统一大小后排成网格拼接"""
    n = len(plotted_list)
    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols
    imgs = [Image.fromarray(img) for img in plotted_list]
    target_w = max(img.width for img in imgs)
    target_h = max(img.height for img in imgs)
    min_sub_size = 1200
    if target_w < min_sub_size or target_h < min_sub_size:
        scale = min_sub_size / min(target_w, target_h)
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)
    imgs = [img.resize((target_w, target_h), Image.LANCZOS) for img in imgs]
    row_imgs = []
    for r in range(rows):
        row = [np.array(imgs[r * cols + c]) for c in range(cols) if r * cols + c < n]
        while len(row) < cols:
            row.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
        row_imgs.append(np.concatenate(row, axis=1))
    montage = np.concatenate(row_imgs, axis=0)
    Image.fromarray(montage).save(save_dir / f'montage_{idx:04d}.jpg', quality=95)


def main():
    if not BEST_PT.exists():
        print(f"[错误] 未找到权重文件: {BEST_PT.resolve()}")
        print("请先完成训练，或检查 runs/detect/train/ 路径。")
        return

    model = YOLO(str(BEST_PT))
    device = pick_device()

    # ========== 1. 测试集定量评估 ==========
    print("=" * 50)
    print("测试集评估结果（全量测试集，按 batch 推理）")
    print("=" * 50)
    print(f"[设备] {device}  |  batch={VAL_BATCH}  imgsz={VAL_IMGSZ}")

    metrics = model.val(
        data=TEST_DATA_YAML,
        split='test',
        imgsz=VAL_IMGSZ,
        batch=VAL_BATCH,
        device=device,
        project='.',
        name='test',
        exist_ok=True,
        save=False,
        plots=False,
        save_txt=False,
        save_conf=False,
        save_json=False,
        max_det=100,
        verbose=False,
    )

    print(f"平均精度(AP50): {metrics.box.map50:.3f}")
    print(f"精确率: {metrics.box.mp:.3f}")
    if hasattr(metrics.box, 'mr'):
        print(f"召回率: {metrics.box.mr:.3f}")

    print("\n" + "=" * 50)
    print("各类别平均精度(AP50):")
    print("=" * 50)

    class_names_cn = {
        'mandatory': '指示标志',
        'prohibitory': '禁止标志',
        'warning': '警告标志',
    }

    for i, name in enumerate(['mandatory', 'prohibitory', 'warning']):
        ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0.0
        print(f"  {class_names_cn[name]}: {ap:.3f}")

    # ========== 2. 释放显存，准备可视化 ==========
    del model, metrics
    torch.cuda.empty_cache()

    # ========== 3. 测试集可视化（生成拼接图） ==========
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    print("\n" + "=" * 50)
    print("测试集可视化 - 生成拼接图")
    print("=" * 50)
    print(f"[加载] 模型: {BEST_PT}")
    model = YOLO(str(BEST_PT))

    test_images = sorted(Path('data/test/images').rglob('*.jpg'))
    random.seed(42)
    random.shuffle(test_images)
    if not test_images:
        print("[跳过] 未找到测试图片")
        return

    total = len(test_images)
    montage_dir = Path(f'./runs/detect/{VIS_RUN_NAME}')
    montage_dir.mkdir(parents=True, exist_ok=True)
    images_per_montage = 9
    print(f"[可视化] 对全部 {total} 张测试图像执行检测，每 {images_per_montage} 张组合为一张拼接图...")

    batch_size = 500
    plotted_buffer = []
    montage_idx = 0

    for i in range(0, total, batch_size):
        batch = test_images[i:i+batch_size]
        print(f"  推理批次 {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} 张)...")
        results = model.predict(
            source=[str(p) for p in batch],
            imgsz=VAL_IMGSZ,
            conf=0.25,
            device=pick_device(),
            save=False,
            project='.',
            name=VIS_RUN_NAME,
            exist_ok=True,
            verbose=False,
        )
        for r in results:
            plotted_buffer.append(r.plot())
        del results
        torch.cuda.empty_cache()

        while len(plotted_buffer) >= images_per_montage:
            montage_idx += 1
            _save_montage(plotted_buffer[:images_per_montage], montage_dir, montage_idx)
            plotted_buffer = plotted_buffer[images_per_montage:]

    if plotted_buffer:
        montage_idx += 1
        _save_montage(plotted_buffer, montage_dir, montage_idx)

    print(f"[✓] 拼接图已保存: {montage_dir}/ (共 {montage_idx} 张拼接图，每张含 {images_per_montage} 个子图)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
