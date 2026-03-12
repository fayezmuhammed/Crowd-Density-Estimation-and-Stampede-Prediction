import torch
from ultralytics import YOLO

def train_fresh():
    assert torch.cuda.is_available(), "CUDA not available. Install a CUDA-enabled PyTorch build."
    torch.cuda.set_device(0)

    model = YOLO("yolov8m.pt")
    model.train(
        data="D:/FYP/jhu.yaml",
        imgsz=2048,
        epochs=200,
        batch=1,
        device=0,
        workers=1,
        nbs=8,
        optimizer='AdamW',
        close_mosaic=0,
    )


    model.val(
        data="D:/FYP/jhu.yaml",
        imgsz=2048,      
        batch=1,
        device=0,
        workers=1,
    )


if _name_ == "_main_":
    train_fresh()