"""
kits_dinov3_deeplab_metrics.py
Adds Dice / IoU evaluation and qualitative visualization.
"""

import os
import argparse
from pathlib import Path
import random
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from transformers import pipeline
import segmentation_models_pytorch as smp
import nibabel as nib
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt



# -------------------------------------------------------------------------
# --- Metrics -------------------------------------------------------------
# -------------------------------------------------------------------------
def dice_coef(pred, target, num_classes):
    """Computes per-class Dice and mean Dice"""
    dice = []
    smooth = 1e-6
    pred_onehot = torch.nn.functional.one_hot(pred, num_classes).permute(0,3,1,2)
    target_onehot = torch.nn.functional.one_hot(target, num_classes).permute(0,3,1,2)
    for c in range(num_classes):
        p = pred_onehot[:,c]
        t = target_onehot[:,c]
        inter = (p*t).sum(dim=(1,2))
        union = p.sum(dim=(1,2)) + t.sum(dim=(1,2))
        d = (2.0*inter + smooth) / (union + smooth)
        dice.append(d.mean().item())
    return dice, np.mean(dice)

def iou_coef(pred, target, num_classes):
    """Computes per-class IoU and mean IoU"""
    ious = []
    smooth = 1e-6
    pred_onehot = torch.nn.functional.one_hot(pred, num_classes).permute(0,3,1,2)
    target_onehot = torch.nn.functional.one_hot(target, num_classes).permute(0,3,1,2)
    for c in range(num_classes):
        p = pred_onehot[:,c]
        t = target_onehot[:,c]
        inter = (p*t).sum(dim=(1,2))
        union = (p + t - p*t).sum(dim=(1,2))
        i = (inter + smooth) / (union + smooth)
        ious.append(i.mean().item())
    return ious, np.mean(ious)

def plot_predictions(imgs, masks, preds, num=3):
    """Shows random sample predictions"""
    b = imgs.shape[0]
    num = min(num, b)
    idxs = np.random.choice(b, num, replace=False)
    for i in idxs:
        img = imgs[i].cpu().permute(1,2,0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        gt = masks[i].cpu().numpy()
        pr = preds[i].cpu().numpy()
        fig, axs = plt.subplots(1,3,figsize=(12,4))
        axs[0].imshow(img); axs[0].set_title("Image")
        axs[1].imshow(gt, cmap="tab20"); axs[1].set_title("GT Mask")
        axs[2].imshow(pr, cmap="tab20"); axs[2].set_title("Prediction")
        for ax in axs: ax.axis('off')
        plt.show()

# -------------------------------------------------------------------------
# --- Simple CT windowing & Dataset --------------------------------------
# -------------------------------------------------------------------------
def window_ct(img, wl=40, ww=400):
    low, high = wl - ww/2, wl + ww/2
    img = np.clip(img, low, high)
    return (img - low) / (high - low)

class KiTSSliceDataset(Dataset):
    def __init__(self, data_dir, transforms=None, slice_step=1):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.images = sorted(glob(str(self.data_dir / "img" / "*.nii*")))
        self.labels = sorted(glob(str(self.data_dir / "mask" / "*.nii*")))
        self.pairs = list(zip(self.images, self.labels))
        self.slice_step = slice_step
        self.samples = []
        for img_path, lbl_path in self.pairs:
            img_nii = nib.load(img_path)
            num_slices = img_nii.shape[2]
            for s in range(0, num_slices, self.slice_step):
                self.samples.append((img_path, lbl_path, s))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, s = self.samples[idx]
        img = nib.load(img_path).get_fdata()[:,:,s]
        lbl = nib.load(lbl_path).get_fdata()[:,:,s]
        img = window_ct(img)
        img = np.stack([img,img,img],axis=-1).astype(np.float32)
        lbl = lbl.astype(np.int64)
        if self.transforms:
            aug = self.transforms(image=img, mask=lbl)
            img, lbl = aug["image"], aug["mask"]
        return {"image": img, "mask": lbl}

# -------------------------------------------------------------------------
# --- DINOv3 feature extractor -------------------------------------------
# -------------------------------------------------------------------------
def build_dino(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", device=0):
    dev = device if torch.cuda.is_available() and device>=0 else -1
    return pipeline("image-feature-extraction", model=model_name, device=dev)

def batch_dino_feats(dino, imgs):
    feats_list=[]
    for i in range(imgs.shape[0]):
        pil = TF.to_pil_image(imgs[i].cpu())
        feats = np.asarray(dino(pil))
        feats = torch.from_numpy(feats).permute(2,0,1).unsqueeze(0).float()
        feats = nn.functional.interpolate(feats, size=imgs.shape[2:], mode='bilinear', align_corners=False)
        feats_list.append(feats)
    return torch.cat(feats_list,0)

# -------------------------------------------------------------------------
# --- Model, validation, training ----------------------------------------
# -------------------------------------------------------------------------
def validate(model, loader, device, dino, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses=[]; dices=[]; ious=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="val"):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            d_feats = batch_dino_feats(dino, imgs.cpu()).to(device)
            inp = torch.cat([imgs, d_feats], 1)
            out = model(inp)
            loss = criterion(out, masks)
            losses.append(loss.item())
            preds = torch.argmax(out,1)
            d_per, d_mean = dice_coef(preds, masks, num_classes)
            i_per, i_mean = iou_coef(preds, masks, num_classes)
            dices.append(d_mean); ious.append(i_mean)
    return {
        "loss": np.mean(losses),
        "dice": np.mean(dices),
        "iou": np.mean(ious)
    }

# -------------------------------------------------------------------------
# --- Main ----------------------------------------------------------------
# -------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = A.Compose([
        A.Resize(384,384),A.HorizontalFlip(p=0.5),A.RandomRotate90(p=0.50),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ToTensorV2()])
    val_tf = A.Compose([
        A.Resize(384,384),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ToTensorV2()])

    ds = KiTSSliceDataset(args.data_dir, transforms=train_tf, slice_step=2)
    n = len(ds)
    tr_idx = int(0.8*n)
    train_ds, val_ds = torch.utils.data.random_split(ds,[tr_idx,n-tr_idx])
    def collate(b): 
        return {"image": torch.stack([x["image"] for x in b]),
                "mask": torch.stack([x["mask"] for x in b])}
    tr_dl = DataLoader(train_ds,batch_size=args.bs,shuffle=True,collate_fn=collate)
    val_dl = DataLoader(val_ds,batch_size=args.bs,shuffle=False,collate_fn=collate)

    print("Loading DINOv3...")
    dino = build_dino(device=0 if torch.cuda.is_available() else -1)
    dummy = next(iter(tr_dl))["image"][:1]
    c_dino = batch_dino_feats(dino,dummy).shape[1]
    print(f"DINO channels {c_dino}")

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50", encoder_weights="imagenet",
        in_channels=3+c_dino, classes=args.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    # --- training loop
    for ep in range(args.epochs):
        model.train()
        tr_losses=[]
        for batch in tqdm(tr_dl,desc=f"train {ep}"):
            imgs=batch["image"].to(device); lbls=batch["mask"].to(device)
            d_feats=batch_dino_feats(dino,imgs.cpu()).to(device)
            inp=torch.cat([imgs,d_feats],1)
            out=model(inp)
            loss=ce(out,lbls)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_losses.append(loss.item())
        valres=validate(model,val_dl,device,dino,args.num_classes)
        print(f"Epoch {ep}: train_loss={np.mean(tr_losses):.3f} | "
              f"val_loss={valres['loss']:.3f} | "
              f"Dice={valres['dice']:.3f} | IoU={valres['iou']:.3f}")
        # visualize few preds
        with torch.no_grad():
            b=next(iter(val_dl))
            imgs=b["image"].to(device)
            lbls=b["mask"].to(device)
            d_feats=batch_dino_feats(dino,imgs.cpu()).to(device)
            preds=torch.argmax(model(torch.cat([imgs,d_feats],1)),1)
            plot_predictions(imgs,lbls,preds,num=2)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir", default=r"C:\Users\Jimmy\Downloads\slices")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--bs",type=int,default=2)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--num_classes",type=int,default=3)
    args=p.parse_args()
    main(args)
