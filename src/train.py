"""Train a crop disease classifier (transfer learning, EfficientNet-B0).

Colab usage:
    !python src/train.py --data /content/PlantVillage --epochs 8
Data layout (torchvision ImageFolder): data/<class_name>/*.jpg
"""
import argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms as T
from sklearn.metrics import classification_report

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--out", default="models/model.pt")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1)), T.RandomHorizontalFlip(),
                      T.RandomRotation(25), T.ColorJitter(0.3, 0.3, 0.2), T.ToTensor(), norm])
eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])

full = datasets.ImageFolder(a.data)
classes = full.classes
n = len(full); n_val = n_test = int(0.15 * n)
g = torch.Generator().manual_seed(42)
tr, va, te = random_split(range(n), [n - n_val - n_test, n_val, n_test], generator=g)

class Sub(torch.utils.data.Dataset):
    def __init__(s, idx, tf): s.idx, s.tf = list(idx), tf
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        img, y = full[s.idx[i]]; return s.tf(img), y

mk = lambda d, sh: DataLoader(d, a.bs, shuffle=sh, num_workers=2)
tl, vl, tel = mk(Sub(tr, train_tf), True), mk(Sub(va, eval_tf), False), mk(Sub(te, eval_tf), False)

m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(classes))
m.to(dev)

def run(loader, opt=None):
    m.train(opt is not None); tot = correct = loss_sum = 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev); out = m(x); loss = nn.functional.cross_entropy(out, y)
            if opt: opt.zero_grad(); loss.backward(); opt.step()
            tot += len(y); correct += (out.argmax(1) == y).sum().item(); loss_sum += loss.item() * len(y)
    return loss_sum / tot, correct / tot

best = 0
for ep in range(a.epochs):
    # phase 1: head only (first 2 epochs), then fine-tune everything
    for q in m.features.parameters(): q.requires_grad = ep >= 2
    opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=1e-3 if ep < 2 else 1e-4)
    trl, tra = run(tl, opt); vll, vla = run(vl)
    print(f"epoch {ep+1}: train {tra:.3f} | val loss {vll:.3f} acc {vla:.3f}")
    if vla > best:
        best = vla; torch.save({"state": m.state_dict(), "classes": classes}, a.out)

m.load_state_dict(torch.load(a.out)["state"]); m.eval(); ys, ps = [], []
with torch.no_grad():
    for x, y in tel: ps += m(x.to(dev)).argmax(1).cpu().tolist(); ys += y.tolist()
print(classification_report(ys, ps, target_names=classes))
