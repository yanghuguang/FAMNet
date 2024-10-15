import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import time
from torchvision import transforms
from torchsummary import summary
from thop import profile

def get_loss():
    return nn.CrossEntropyLoss()

def compute_metrics(pred, target, num_classes=5):
    pred = pred.view(-1)
    target = target.view(-1)

    MIoU = 0
    MA = 0
    PA = 0
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            iou = float('nan') 
        else:
            iou = float(intersection) / float(max(union, 1))
        MIoU += iou

        if target_inds.long().sum().item() == 0:
            acc = float('nan')
        else:
            acc = float(intersection) / float(max(target_inds.long().sum().item(), 1))
        MA += acc

        PA += float(intersection)

        total = target_inds.long().sum().item()

    MIoU /= num_classes
    MA /= num_classes
    PA /= (target.numel())
    return MIoU, MA, PA

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")
    return total_params

def calculate_flops(model, input_size=(3, 512, 512)):
    input = torch.randn(1, *input_size)
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    return flops

def measure_fps(model, input_size=(1, 3, 512, 512), iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    input = torch.randn(*input_size).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(input)

    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input)
    end = time.time()
    avg_time = (end - start) / iterations
    fps = 1 / avg_time
    print(f"Average Inference Time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    return fps

from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            mask = torch.squeeze(mask)  

        return image, mask
