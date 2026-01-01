from __future__ import annotations
import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score

from utils import seed_everything, ensure_dir
from dataset_mura import build_image_samples, MuraImageDataset


def make_model(num_classes: int = 1) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs = []
    ys = []

    for x, y, _paths in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        p = torch.sigmoid(logits)
        probs.append(p.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())

    probs = np.concatenate(probs, axis=0).reshape(-1)
    ys = np.concatenate(ys, axis=0).reshape(-1)

    # metrics
    # AUROC requires both classes present; otherwise handle gracefully
    auroc = None
    try:
        auroc = float(roc_auc_score(ys, probs))
    except Exception:
        auroc = None

    preds = (probs >= 0.5).astype(np.int32)
    cm = confusion_matrix(ys.astype(np.int32), preds, labels=[0, 1])
    prec = float(precision_score(ys, preds, zero_division=0))
    rec = float(recall_score(ys, preds, zero_division=0))
    f1 = float(f1_score(ys, preds, zero_division=0))

    return {"auroc": auroc, "cm": cm, "precision": prec, "recall": rec, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mura_root", type=str, required=True, help="Path to MURA-v1.1 folder")
    parser.add_argument("--train_csv", type=str, default="train_labeled_studies.csv")
    parser.add_argument("--valid_csv", type=str, default="valid_labeled_studies.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    valid_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Build samples from MURA CSVs
    train_samples = build_image_samples(args.mura_root, args.train_csv)
    valid_samples = build_image_samples(args.mura_root, args.valid_csv)

    train_ds = MuraImageDataset(train_samples, transform=train_tf)
    valid_ds = MuraImageDataset(valid_samples, transform=valid_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = make_model().to(device)

    # handle imbalance with pos_weight
    y_train = np.array([s.label for s in train_samples], dtype=np.int32)
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -1.0
    best_path = os.path.join(args.model_dir, "best_resnet18_mura.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, y, _paths in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            pbar.set_postfix(loss=np.mean(losses))

        scheduler.step()

        metrics = evaluate(model, valid_loader, device)
        auroc = metrics["auroc"]
        print(f"Epoch {epoch}: train_loss={np.mean(losses):.4f}  "
              f"val_auroc={auroc}  val_f1={metrics['f1']:.4f}  "
              f"val_prec={metrics['precision']:.4f}  val_rec={metrics['recall']:.4f}")

        # choose best by AUROC if available, otherwise by F1
        score = auroc if auroc is not None else metrics["f1"]
        if score > best_score:
            best_score = score
            torch.save({
                "model_state": model.state_dict(),
                "img_size": args.img_size,
                "arch": "resnet18",
            }, best_path)
            print(f"Saved best model -> {best_path}")
            print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
            print(metrics["cm"])

    print(f"Done. Best score: {best_score:.4f}")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()
