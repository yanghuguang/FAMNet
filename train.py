import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from FAMNet import FAMNet
from utils import get_loss, compute_metrics, CustomDataset
import argparse
import os
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

def get_dataloader(images_dir, masks_dir, batch_size, num_workers, train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                 std=[0.229, 0.224, 0.225])   
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    target_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    dataset = CustomDataset(images_dir=images_dir, masks_dir=masks_dir,
                            transform=transform, target_transform=target_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    return dataloader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FAMNet(backbone=args.backbone, num_classes=args.num_classes, pretrained=True, d=args.d).to(device)
    criterion = get_loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = get_dataloader(images_dir=args.train_images,
                                  masks_dir=args.train_masks,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  train=True)
    val_loader = get_dataloader(images_dir=args.val_images,
                                masks_dir=args.val_masks,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                train=False)

    best_miou = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            masks = masks.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

        model.eval()
        miou_total = 0
        ma_total = 0
        pa_total = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device).long()
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                miou, ma, pa = compute_metrics(preds, masks, num_classes=args.num_classes)
                miou_total += miou
                ma_total += ma
                pa_total += pa

        avg_miou = miou_total / len(val_loader)
        avg_ma = ma_total / len(val_loader)
        avg_pa = pa_total / len(val_loader)
        print(f"Validation MIoU: {avg_miou:.4f}, MA: {avg_ma:.4f}, PA: {avg_pa:.4f}")

        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print("Best model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FAMNet for Semantic Segmentation")
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet50', 'resnet101'], help='Backbone network')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--train_images', type=str, required=True, help='Path to training images directory')
    parser.add_argument('--train_masks', type=str, required=True, help='Path to training masks directory')
    parser.add_argument('--val_images', type=str, required=True, help='Path to validation images directory')
    parser.add_argument('--val_masks', type=str, required=True, help='Path to validation masks directory')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of segmentation classes (including background)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--d', type=int, default=256, help='Dimension for FAM module')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
