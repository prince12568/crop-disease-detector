"""Evaluate a model on the HELD-OUT PlantDoc test split (images the model never trained on).

Colab (run from the project root):
    !pip install -q datasets scikit-learn matplotlib
    !python src/eval_realworld.py --model models/model_v2.pt
    !python src/eval_realworld.py --model models/model.pt --out eval_outputs/v1   # optional v1 baseline
Uses eval_outputs/plantdoc_split.json from train_v2.py; if that file is missing it recreates the
same split (same seed). Outputs: results.txt, confusion_matrix.png, gradcam_examples.png
"""
import argparse, json, os, random
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision import models, transforms as T

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/model_v2.pt")
p.add_argument("--split", default="eval_outputs/plantdoc_split.json")
p.add_argument("--out", default="eval_outputs/v2")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

MAP = {  # PlantDoc class -> your model's class (same mapping as train_v2.py)
    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot", "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato leaf late blight": "Tomato_Late_blight", "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus", "Tomato leaf": "Tomato_healthy",
    "Potato leaf early blight": "Potato___Early_blight", "Potato leaf late blight": "Potato___Late_blight",
    "Bell_pepper leaf": "Pepper__bell___healthy", "Bell_pepper leaf spot": "Pepper__bell___Bacterial_spot",
}
ckpt = torch.load(a.model, map_location=dev)
classes = ckpt["classes"]
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(ckpt["state"]); model.to(dev).eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ds = load_dataset("Project-AgML/plant_doc_classification")["train"]
if os.path.exists(a.split):
    test = json.load(open(a.split))["test"]
else:  # recreate the exact split train_v2.py used
    names = ds.features["label"].names
    items = [(i, classes.index(MAP[names[l]])) for i, l in enumerate(ds["label"]) if names[l] in MAP]
    ys = [y for _, y in items]
    _, tmp = train_test_split(range(len(items)), test_size=0.4, stratify=ys, random_state=42)
    _, te = train_test_split(tmp, test_size=0.5, stratify=[ys[k] for k in tmp], random_state=42)
    test = [items[k] for k in te]
print(f"Model: {a.model} | held-out test images: {len(test)}")

y_true, y_pred, conf, top3 = [], [], [], []
with torch.no_grad():
    for j, y in test:
        img = ds[j]["image"].convert("RGB")
        pr = model(tf(img).unsqueeze(0).to(dev)).softmax(1)[0].cpu()
        t = pr.topk(3).indices.tolist()
        y_true.append(y); y_pred.append(t[0]); conf.append(pr[t[0]].item()); top3.append(y in t)
y_true, y_pred, conf = np.array(y_true), np.array(y_pred), np.array(conf)
correct = y_true == y_pred

# ---- metrics + confidence-threshold table (use it to choose the app's warning threshold) ----
labels = sorted(set(y_true.tolist()))
report = classification_report(y_true, y_pred, labels=labels, zero_division=0,
                               target_names=[classes[k] for k in labels])
rows = ["threshold  coverage  accuracy_on_answered"]
for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    m = conf >= th
    rows.append(f"{th:>9.1f}  {m.mean():>8.2f}  {correct[m].mean() if m.any() else float('nan'):>20.3f}")
summary = (f"Model: {a.model}\nImages tested (held-out): {len(y_true)}\n"
           f"Top-1 accuracy: {correct.mean():.3f}\nTop-3 accuracy: {np.mean(top3):.3f}\n"
           f"Mean confidence: {conf.mean():.3f}\n\n{report}\n"
           "Confidence threshold: share of images answered vs accuracy on those\n" + "\n".join(rows))
print(summary); open(f"{a.out}/results.txt", "w").write(summary)

# ---- confusion matrix ----
lab = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
cm = confusion_matrix(y_true, y_pred, labels=lab)
fig, ax = plt.subplots(figsize=(11, 9)); ax.imshow(cm, cmap="Greens")
ticks = [classes[k] for k in lab]
ax.set_xticks(range(len(lab))); ax.set_xticklabels(ticks, rotation=90, fontsize=7)
ax.set_yticks(range(len(lab))); ax.set_yticklabels(ticks, fontsize=7)
for r in range(len(lab)):
    for c in range(len(lab)):
        if cm[r, c]: ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=7)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Held-out PlantDoc confusion matrix")
plt.tight_layout(); plt.savefig(f"{a.out}/confusion_matrix.png", dpi=150); plt.close()

# ---- Grad-CAM examples ----
def cam_overlay(img):
    acts, grads = [], []
    def fwd(_, __, o): acts.append(o); o.register_hook(grads.append)
    h = model.features[-1].register_forward_hook(fwd)
    out = model(tf(img).unsqueeze(0).to(dev)); model.zero_grad()
    out[0, out.argmax()].backward(); h.remove()
    w = grads[0].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts[0]).sum(1))[0].detach().cpu().numpy()
    cam = cv2.resize((cam - cam.min()) / (cam.max() - cam.min() + 1e-8), (224, 224))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(np.array(img.resize((224, 224))), 0.55, heat, 0.45, 0)

random.seed(0)
ok, bad = np.where(correct)[0].tolist(), np.where(~correct)[0].tolist()
picks = random.sample(ok, min(4, len(ok))) + random.sample(bad, min(4, len(bad)))
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, k in zip(axes.flat, picks):
    ax.imshow(cam_overlay(ds[test[k][0]]["image"].convert("RGB"))); ax.axis("off")
    ax.set_title(f"true: {classes[y_true[k]]}\npred: {classes[y_pred[k]]} ({conf[k]:.0%})", fontsize=7,
                 color="green" if correct[k] else "red")
for ax in axes.flat[len(picks):]: ax.axis("off")
plt.tight_layout(); plt.savefig(f"{a.out}/gradcam_examples.png", dpi=150); plt.close()
print(f"\nSaved results to {a.out}/")
