import base64, io
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms as T

MODEL_PATH, MAX_MB, MIN_CONF = "models/model.pt", 8, 0.6
ckpt = torch.load(MODEL_PATH, map_location="cpu")
classes = ckpt["classes"]
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(ckpt["state"]); model.eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

app = FastAPI(title="Crop Disease Detector")

def predict_with_cam(img: Image.Image):
    x, acts, grads = tf(img).unsqueeze(0), [], []
    def fwd(_, __, o): acts.append(o); o.register_hook(grads.append)
    h = model.features[-1].register_forward_hook(fwd)
    out = model(x); probs = out.softmax(1)[0].detach(); top = probs.topk(3)
    model.zero_grad(); out[0, top.indices[0]].backward(); h.remove()
    w = grads[0].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts[0]).sum(1))[0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return top, cam

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Upload a JPG or PNG image.")
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_MB} MB.")
    try: img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception: raise HTTPException(400, "Could not read this image.")
    top, cam = predict_with_cam(img)
    rgb = np.array(img)
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, rgb.shape[1::-1])), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.55, heat, 0.45, 0)
    png = base64.b64encode(cv2.imencode(".png", overlay)[1]).decode()
    preds = [{"label": classes[i], "confidence": round(float(p), 4)} for p, i in zip(top.values, top.indices)]
    return {"predictions": preds, "low_confidence": preds[0]["confidence"] < MIN_CONF,
            "heatmap": f"data:image/png;base64,{png}"}

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
