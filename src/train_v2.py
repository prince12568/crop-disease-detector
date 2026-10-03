"""v2 training: PlantVillage + real PlantDoc photos, stronger augmentation.

Colab (run from the project root):
    !pip install -q datasets scikit-learn
    !python src/train_v2.py --pv /content/data/PlantVillage --init models/model.pt
Trains on PlantVillage (same train split as v1) + 60% of the mapped PlantDoc images,
picks the best epoch on a PlantDoc validation split (20%), and reports on a held-out
PlantDoc test split (20%) that is never trained on. It also scores your v1 model on the
same test split so you get a fair before/after comparison.
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import datasets, models, transforms as T
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

p = argparse.ArgumentParser()
p.add_argument("--pv", required=True, help="PlantVillage ImageFolder path (same one used for v1)")
p.add_argument("--init", default=None, help="start from this checkpoint (e.g. models/model.pt)")
p.add_argument("--v1", default="models/model.pt", help="v1 checkpoint for the comparison")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--per_epoch", type=int, default=8000, help="samples drawn per epoch")
p.add_argument("--out", default="models/model_v2.pt")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("eval_outputs", exist_ok=True)

MAP = {  # PlantDoc class -> your model's class (same mapping as eval_realworld.py)
    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot", "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato leaf late blight": "Tomato_Late_blight", "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus", "Tomato leaf": "Tomato_healthy",
    "Potato leaf early blight": "Potato___Early_blight", "Potato leaf late blight": "Potato___Late_blight",
    "Bell_pepper leaf": "Pepper__bell___healthy", "Bell_pepper leaf spot": "Pepper__bell___Bacterial_spot",
}
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.3, 1)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.RandomRotation(30), T.ColorJitter(0.4, 0.4, 0.4, 0.05), T.RandomGrayscale(0.05),
                      T.GaussianBlur(5, (0.1, 2.0)), T.ToTensor(), norm, T.RandomErasing(0.25)])
eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])

# ---- PlantVillage: identical train split to v1 (same seed) ----
full = datasets.ImageFolder(a.pv); classes = full.classes
n = len(full); n_val = n_test = int(0.15 * n)
pv_train_idx, _, _ = random_split(range(n), [n - n_val - n_test, n_val, n_test],
                                  generator=torch.Generator().manual_seed(42))
# ---- PlantDoc: stratified 60/20/20 split, saved to disk ----
ds = load_dataset("Project-AgML/plant_doc_classification")["train"]
names = ds.features["label"].names
items = [(i, classes.index(MAP[names[l]])) for i, l in enumerate(ds["label"]) if names[l] in MAP]
ys = [y for _, y in items]
tr_i, tmp_i = train_test_split(range(len(items)), test_size=0.4, stratify=ys, random_state=42)
va_i, te_i = train_test_split(tmp_i, test_size=0.5, stratify=[ys[k] for k in tmp_i], random_state=42)
pd_tr, pd_va, pd_te = [[items[k] for k in s] for s in (tr_i, va_i, te_i)]
json.dump({"train": pd_tr, "val": pd_va, "test": pd_te}, open("eval_outputs/plantdoc_split.json", "w"))
print(f"PlantVillage train {len(pv_train_idx)} | PlantDoc train {len(pd_tr)} val {len(pd_va)} test {len(pd_te)}")

class PV(Dataset):
    def __init__(s, idx, tf): s.idx, s.tf = list(idx), tf
    def __len__(s): return len(s.idx)
    def __getitem__(s, i): img, y = full[s.idx[i]]; return s.tf(img), y

class PD(Dataset):
    def __init__(s, its, tf): s.its, s.tf = its, tf
    def __len__(s): return len(s.its)
    def __getitem__(s, i): j, y = s.its[i]; return s.tf(ds[j]["image"].convert("RGB")), y

# half of every epoch comes from real PlantDoc photos, half from PlantVillage
pv_tr, pd_train = PV(pv_train_idx, train_tf), PD(pd_tr, train_tf)
w = [0.5 / len(pv_tr)] * len(pv_tr) + [0.5 / len(pd_train)] * len(pd_train)
tl = DataLoader(ConcatDataset([pv_tr, pd_train]), a.bs, num_workers=2,
                sampler=WeightedRandomSampler(w, a.per_epoch, replacement=True))
vl = DataLoader(PD(pd_va, eval_tf), a.bs, num_workers=2)
tel = DataLoader(PD(pd_te, eval_tf), a.bs, num_workers=2)

def build(path=None):
    m = models.efficientnet_b0(weights=None if path else models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(classes))
    if path: m.load_state_dict(torch.load(path, map_location=dev)["state"])
    return m.to(dev)

def predict(m, loader):
    m.eval(); y, pr, cf = [], [], []
    with torch.no_grad():
        for x, t in loader:
            c, k = m(x.to(dev)).softmax(1).cpu().max(1); y += t.tolist(); pr += k.tolist(); cf += c.tolist()
    return np.array(y), np.array(pr), np.array(cf)

m = build(a.init)
opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
best = 0
for ep in range(a.epochs):
    m.train(); loss_sum = 0
    for x, y in tl:
        x, y = x.to(dev), y.to(dev)
        loss = nn.functional.cross_entropy(m(x), y, label_smoothing=0.1)  # label smoothing curbs overconfidence
        opt.zero_grad(); loss.backward(); opt.step(); loss_sum += loss.item()
    sched.step()
    yv, pv_, _ = predict(m, vl); acc = (yv == pv_).mean()
    print(f"epoch {ep+1}: train loss {loss_sum/len(tl):.3f} | PlantDoc val acc {acc:.3f}")
    if acc > best: best = acc; torch.save({"state": m.state_dict(), "classes": classes}, a.out)

# ---- final comparison on the held-out PlantDoc test split ----
labels = sorted(set(y for _, y in pd_te))
def score(model, name):
    y, pr, cf = predict(model, tel)
    print(f"\n{name}: top-1 acc {(y == pr).mean():.3f} | mean confidence {cf.mean():.3f}")
    return y, pr
if os.path.exists(a.v1): score(build(a.v1), "v1 (PlantVillage only)")
y, pr = score(build(a.out), "v2 (PlantVillage + PlantDoc)")
print(classification_report(y, pr, labels=labels, zero_division=0, target_names=[classes[k] for k in labels]))
