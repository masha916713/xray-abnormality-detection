### نصب
pip install -r requirements.txt

### آموزش (برای MURA)
python src/train.py --mura_root data/MURA-v1.1 --epochs 8 --batch_size 32

### پیش‌بینی روی یک تصویر
python src/predict.py --ckpt models/best_resnet18_mura.pt --image path/to/image.png

### تولید Grad-CAM
python src/gradcam.py --ckpt models/best_resnet18_mura.pt --image path/to/image.png --out outputs/gradcam.png
