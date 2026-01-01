from __future__ import annotations
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_m, _inp, out):
            self.activations = out.detach()

        def bwd_hook(_m, _gin, gout):
            self.gradients = gout[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                 # [1,1]
        score = logits[0, 0]
        score.backward(retain_graph=False)

        grads = self.gradients[0]             # [C,H,W]
        acts = self.activations[0]            # [C,H,W]
        weights = grads.mean(dim=(1, 2))      # [C]

        cam = (weights[:, None, None] * acts).sum(dim=0)  # [H,W]
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-6)
        return cam.cpu().numpy()

def make_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/gradcam.png")
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

    # target layer for ResNet18
    target_layer = model.layer4[-1].conv2
    cam = GradCAM(model, target_layer)

    pil = Image.open(args.image).convert("RGB")
    x = tf(pil).unsqueeze(0)

    heat = cam(x)  # [H,W] in 0..1

    # overlay
    img = np.array(pil.resize((img_size, img_size)))
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img[:, :, ::-1], 0.6, heat_color, 0.4, 0)  # BGR

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, overlay)
    print(f"Saved Grad-CAM overlay -> {args.out}")

if __name__ == "__main__":
    import os
    main()
