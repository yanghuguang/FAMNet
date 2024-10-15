
import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessing for Semantic Segmentation")
    parser.add_argument('--images_dir', type=str, required=True, help='Path to the original images directory')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to the original masks directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the processed dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Proportion of training data')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Proportion of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Proportion of test data')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation to training set')
    args = parser.parse_args()
    return args

def get_filenames(images_dir, masks_dir):
    images = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
    masks = sorted([f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))])
    assert len(images) == len(masks), "Number of images and masks should be equal"
    for img, msk in zip(images, masks):
        img_base = os.path.splitext(img)[0]
        msk_base = os.path.splitext(msk)[0]
        assert img_base == msk_base, f"Image and mask names do not match: {img} vs {msk}"
    return images

def split_data(filenames, train_ratio, val_ratio, test_ratio):
    train_val_ratio = train_ratio + val_ratio
    train_val, test = train_test_split(filenames, test_size=test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_val_ratio), random_state=42)
    return train, val, test

def create_dirs(output_dir):
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        img_dir = os.path.join(output_dir, subset, 'images')
        msk_dir = os.path.join(output_dir, subset, 'masks')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

def copy_files(filenames, src_images_dir, src_masks_dir, dst_images_dir, dst_masks_dir, augment=False):
    if augment:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])
    else:
        transform = None

    for filename in filenames:
        img_path = os.path.join(src_images_dir, filename)
        msk_path = os.path.join(src_masks_dir, filename)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(msk_path).convert("L"))

        if transform:
            augmented = transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        img_save_path = os.path.join(dst_images_dir, filename)
        msk_save_path = os.path.join(dst_masks_dir, filename)

        Image.fromarray(image).save(img_save_path)
        Image.fromarray(mask).save(msk_save_path)



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

def main():
    args = parse_args()

    filenames = get_filenames(args.images_dir, args.masks_dir)
    print(f"Total samples found: {len(filenames)}")

    train_filenames, val_filenames, test_filenames = split_data(filenames, args.train_ratio, args.val_ratio, args.test_ratio)
    print(f"Training samples: {len(train_filenames)}")
    print(f"Validation samples: {len(val_filenames)}")
    print(f"Test samples: {len(test_filenames)}")
    create_dirs(args.output_dir)

    print("Processing training data...")
    copy_files(train_filenames, args.images_dir, args.masks_dir,
               os.path.join(args.output_dir, 'train', 'images'),
               os.path.join(args.output_dir, 'train', 'masks'),
               augment=args.augment)

    print("Processing validation data...")
    copy_files(val_filenames, args.images_dir, args.masks_dir,
               os.path.join(args.output_dir, 'val', 'images'),
               os.path.join(args.output_dir, 'val', 'masks'),
               augment=False)

    print("Processing test data...")
    copy_files(test_filenames, args.images_dir, args.masks_dir,
               os.path.join(args.output_dir, 'test', 'images'),
               os.path.join(args.output_dir, 'test', 'masks'),
               augment=False)

    print("Data processing completed successfully.")

if __name__ == "__main__":
    main()
