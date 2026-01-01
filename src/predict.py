from __future__ import annotations
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

def make_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    img_size = int(ckpt.get("img_size", 224))

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model = make_model()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0)  # [1,3,H,W]

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    pred = "ABNORMAL (مشکوک/غیرطبیعی)" if prob >= 0.5 else "NORMAL (طبیعی)"
    print(f"prob_abnormal={prob:.4f}  =>  {pred}")

if __name__ == "__main__":
    main()
