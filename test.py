import torch
from FAMNet import FAMNet
from utils import compute_metrics, CustomDataset
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

def get_dataloader(images_dir, masks_dir, batch_size, num_workers):
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FAMNet(backbone=args.backbone, num_classes=args.num_classes, pretrained=False, d=args.d).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    test_loader = get_dataloader(images_dir=args.test_images,
                                 masks_dir=args.test_masks,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    miou_total = 0
    ma_total = 0
    pa_total = 0
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device).long()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            miou, ma, pa = compute_metrics(preds, masks, num_classes=args.num_classes)
            miou_total += miou
            ma_total += ma
            pa_total += pa

    avg_miou = miou_total / len(test_loader)
    avg_ma = ma_total / len(test_loader)
    avg_pa = pa_total / len(test_loader)
    print(f"Test MIoU: {avg_miou:.4f}, MA: {avg_ma:.4f}, PA: {avg_pa:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FAMNet for Semantic Segmentation")
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet50', 'resnet101'], help='Backbone network')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--test_images', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--test_masks', type=str, required=True, help='Path to test masks directory')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of segmentation classes (including background)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--d', type=int, default=256, help='Dimension for FAM module')
    args = parser.parse_args()

    test(args)
