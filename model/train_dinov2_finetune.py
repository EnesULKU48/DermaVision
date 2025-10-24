# =====================================================
# DINOv2 SWEATY FINETUNE — RTX 4060 optimized
# =====================================================
import os, random, numpy as np, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------ reproducibility ------------------
def seed_all(s=42):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
seed_all(42)

# ------------------ EMA (simple) ---------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

# ------------------ data -----------------------------
def make_loaders(root="dataset", img_size=256, batch=8):
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    mean = getattr(proc, "image_mean", [0.485,0.456,0.406])
    std  = getattr(proc, "image_std",  [0.229,0.224,0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(img_size+32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_train_all = datasets.ImageFolder(root, transform=train_tf)
    ds_val_all   = datasets.ImageFolder(root, transform=val_tf)
    classes = ds_train_all.classes

    # ---- stratified 80/20 split ----
    paths, targets = zip(*ds_train_all.samples)
    targets = np.array(targets)
    idx_per_class = [np.where(targets==c)[0] for c in range(len(classes))]
    train_idx, val_idx = [], []
    for arr in idx_per_class:
        n = len(arr); n_train = int(0.8*n)
        perm = np.random.permutation(arr)
        train_idx.extend(perm[:n_train]); val_idx.extend(perm[n_train:])
    train_ds = Subset(ds_train_all, train_idx)
    val_ds   = Subset(ds_val_all,   val_idx)

    # raporla
    def count(ds, base):
        cnt=[0]*len(classes)
        for i in ds.indices: _, y = base.samples[i]; cnt[y]+=1
        return dict(zip(classes, cnt))
    print("📦 Train dağılımı:", count(train_ds, ds_train_all))
    print("📦 Val dağılımı  :", count(val_ds, ds_train_all))

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
    return train_dl, val_dl, classes

# ------------------ model ----------------------------
class DinoHead(nn.Module):
    def __init__(self, in_dim=768, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,256), nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256,n_classes)
        )
    def forward(self, x): return self.net(x)

class DinoModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.base = AutoModel.from_pretrained("facebook/dinov2-base")
        self.head = DinoHead(768, n_classes)
    def forward(self, x):
        feats = self.base(x).last_hidden_state[:,0]
        return self.head(feats)

# ------------- param groups & unfreeze ---------------
def set_stage(model, stage):
    """
    stage 0: only head train
    stage 1: unfreeze layers.11 + norm
    stage 2: unfreeze layers.10 as well
    """
    for p in model.base.parameters(): p.requires_grad=False
    if stage>=1:
        for n,p in model.base.named_parameters():
            if "layers.11" in n or "norm" in n:
                p.requires_grad=True
    if stage>=2:
        for n,p in model.base.named_parameters():
            if "layers.10" in n or "layers.11" in n or "norm" in n:
                p.requires_grad=True

def make_optimizer(model, stage):
    # discriminative LR: head fastest, backbone slower
    params=[]
    head_lr = {0:1e-3, 1:7e-4, 2:5e-4}[stage]
    bb_top_lr = {0:0.0, 1:2e-4, 2:1e-4}[stage]
    # head
    params.append({"params": model.head.parameters(), "lr": head_lr, "weight_decay":1e-5})
    # backbone trainables
    pg=[]
    for n,p in model.base.named_parameters():
        if p.requires_grad: pg.append(p)
    if pg: params.append({"params": pg, "lr": bb_top_lr, "weight_decay":1e-5})
    return optim.AdamW(params)

def cosine_warmup(step, total, warmup=0.05):
    if step < total*warmup:
        return step/(total*warmup)
    t = (step - total*warmup)/(total*(1-warmup))
    return 0.5*(1+np.cos(np.pi*float(t)))

# ------------------ train ----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🟢 Device:", device)
    scaler = torch.amp.GradScaler('cuda')

    IMG=256; BATCH=8; EPOCHS=30
    train_dl, val_dl, classes = make_loaders("dataset", IMG, BATCH)

    model = DinoModel(len(classes)).to(device)

    # label smoothing helps a lot on small data
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc=0.0; patience=8; wait=0
    tr_losses=[]; va_losses=[]; va_accs=[]
    ema = EMA(model, decay=0.999)

    # --- staged training plan ---
    plan = [(0,5), (1,10), (2,EPOCHS)]  # (stage, until_epoch)
    stage=0; set_stage(model, stage)
    optimizer = make_optimizer(model, stage)

    total_steps = len(train_dl)*EPOCHS; step=0

    for epoch in range(1, EPOCHS+1):
        # stage switch
        for st, until in plan:
            if epoch==until+1 and stage<st:
                stage=st; set_stage(model, stage)
                optimizer = make_optimizer(model, stage)
                print(f"🔁 Stage -> {stage} (unfreeze policy changed)")

        # -------- train --------
        model.train()
        run=0.0
        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # cosine warmup lr
            scale = cosine_warmup(step, total_steps, warmup=0.06)
            for g in optimizer.param_groups:
                g["lr"] = g["lr"]*0 + g["lr"]  # keep base
            scaler.step(optimizer); scaler.update()
            ema.update(model)
            run += loss.item(); step+=1
        tr_loss = run/len(train_dl)

        # -------- validate (EMA weights) --------
        backup = {k: v.clone() for k,v in model.state_dict().items()}
        ema.copy_to(model);  # eval with EMA
        model.eval(); correct=total=0; vloss=0.0
        all_preds=[]; all_labels=[]
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)
                correct += (preds==labels).sum().item()
                total += labels.size(0)
                vloss += loss.item()
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
        acc = correct/total; va_loss = vloss/len(val_dl)
        # restore training weights
        model.load_state_dict(backup)

        tr_losses.append(tr_loss); va_losses.append(va_loss); va_accs.append(acc)
        print(f"🎯 Epoch {epoch}: TrainLoss={tr_loss:.4f} | ValLoss={va_loss:.4f} | ValAcc={acc:.3f}")

        if acc>best_acc:
            best_acc=acc; wait=0
            # save EMA weights as the "best"
            ema.copy_to(model)
            torch.save(model.state_dict(), "best_dino_sweaty.pth")
            model.load_state_dict(backup)
        else:   
            wait+=1
            if wait>=patience:
                print("⛔ Early stop."); break

    print(f"✅ Best ValAcc: {best_acc:.3f}")
    print("\n📊 Report (last epoch, EMA):")
    ema.copy_to(model)
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

    # plots
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(tr_losses,label="Train"); plt.plot(va_losses,label="Val"); plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2); plt.plot(va_accs,label="ValAcc"); plt.legend(); plt.title("Accuracy"); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()
