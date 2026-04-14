
import os
import sys
import cv2
import torch
import numpy as np
import sys
sys.path.append("E:/paper/class5/loss")
sys.path.append("E:/paper/class5/models")
sys.path.append("E:/paper/class5/dataset")
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from lossupdate import HybridLossV2
from custom_dataset import DeepLabSegmentationDataset as SegmentationDataset
from DeepLabV3Plus_MSAM import DeepLabV3Plus_MSAM


# ========= 类别颜色映射 =========
color_map = {
    0: (0, 0, 0),
    1: (239, 41, 41),
    2: (252, 175, 62),
    3: (252, 233, 79),
    4: (81, 255, 0)
}

# ========= 可视化输出函数 =========
def save_pred_color(preds, names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    preds = preds.argmax(dim=1).cpu().numpy()
    for i in range(len(preds)):
        pred = preds[i]
        color_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for cls_id, color in color_map.items():
            color_mask[pred == cls_id] = color
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"{names[i]}.png"), color_mask)

# ========= 精确指标统计 =========
def compute_metrics_v2(preds, labels, num_classes):
    preds = preds.argmax(dim=1)
    metrics = {
        "intersection": np.zeros(num_classes),
        "union": np.zeros(num_classes),
        "pred": np.zeros(num_classes),
        "label": np.zeros(num_classes)
    }

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        pred_sum = pred_cls.sum().item()
        label_sum = label_cls.sum().item()
        metrics["intersection"][cls] += intersection
        metrics["union"][cls] += union
        metrics["pred"][cls] += pred_sum
        metrics["label"][cls] += label_sum

    return metrics

# ========= 主训练函数 =========
def train_model(config, resume_epoch):
    model = DeepLabV3Plus_MSAM(n_classes=config["num_classes"]).to(config["device"])
    best_model_path = os.path.join(config["save_path"], "best.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("🔁 加载之前最佳模型继续训练")

    train_dataset = SegmentationDataset(config["img_dir"], config["mask_dir"], config["train_list"])
    val_dataset = SegmentationDataset(config["img_dir"], config["mask_dir"], config["val_list"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    criterion = HybridLossV2(
        losses=["cbfocal", "tversky"],
        weights=[0.4, 0.6],
        loss_args=[
            {"class_freq": [0.039, 3.9326, 0.9, 1.1552, 0.9], "ignore_index": 255},
            {"alpha": 0.7, "beta": 0.3, "ignore_index": 255}
        ]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    best_miou = 0.0
    patience = config.get("early_stopping_patience", 7)
    no_improve_epochs = 0

    for epoch in range(resume_epoch, resume_epoch + config["epochs"]):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}]")
        for imgs, masks, _ in loop:
            imgs, masks = imgs.to(config["device"]), masks.to(config["device"])
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        # ========= 验证阶段 =========
        model.eval()
        num_classes = config["num_classes"]
        total = {
            "intersection": np.zeros(num_classes),
            "union": np.zeros(num_classes),
            "pred": np.zeros(num_classes),
            "label": np.zeros(num_classes)
        }

        save_pred_dir = os.path.join("E:/paper/class5/vis", f"epoch{epoch+1}")
        os.makedirs(save_pred_dir, exist_ok=True)

        with torch.no_grad():
            for imgs, masks, names in val_loader:
                imgs, masks = imgs.to(config["device"]), masks.to(config["device"])
                preds = model(imgs)
                m = compute_metrics_v2(preds, masks, num_classes)
                for k in total:
                    total[k] += m[k]
                save_pred_color(preds, names, save_pred_dir)

        acc = total["intersection"] / (total["label"] + 1e-6)
        iou = total["intersection"] / (total["union"] + 1e-6)
        dice = 2 * total["intersection"] / (total["label"] + total["pred"] + 1e-6)
        miou = np.mean(iou)

        print(f"[VAL] mIoU={miou:.4f}")
        class_names = ["background", "person", "building", "car", "tree"]
        print("[VAL] Per-Class Metrics:")
        for i, name in enumerate(class_names):
            print(f"  {name:<10} | Acc: {acc[i]:.4f} | IoU: {iou[i]:.4f} | Dice: {dice[i]:.4f}")

        with open("log.txt", "a") as f:
            f.write(f"{epoch + 1},{avg_loss:.4f},{miou:.4f},")
            f.write(",".join(f"{x:.4f}" for x in acc) + ",")
            f.write(",".join(f"{x:.4f}" for x in iou) + ",")
            f.write(",".join(f"{x:.4f}" for x in dice) + "\n")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), best_model_path)
            print("✅ Saved best model.")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"⚠️ mIoU 未提升，连续 {no_improve_epochs}/{patience} 轮无提升")

        if no_improve_epochs >= patience:
            print("🛑 Early stopping triggered.")
            break

        scheduler.step()

# ========= 启动入口 =========
if __name__ == "__main__":
    config = {
        "img_dir": "E:/paper/class5/patch/image",
        "mask_dir": "E:/paper/class5/patch/mask",
        "train_list": "E:/paper/class5/patch/train_list.txt",
        "val_list": "E:/paper/class5/patch/val_list.txt",
        "num_classes": 5,
        "batch_size": 4,
        "lr": 1e-4,
        "epochs": 10,
        "save_path": "E:/paper/class5/checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "early_stopping_patience": 7
    }

    resume_epoch = 0
    if os.path.exists("log.txt"):
        with open("log.txt", "r") as f:
            lines = f.readlines()
            if lines:
                try:
                    resume_epoch = int(lines[-1].strip().split(",")[0])
                except:
                    resume_epoch = 0

    train_model(config, resume_epoch)
