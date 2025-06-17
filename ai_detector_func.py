import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button

# === 載入模型 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load("resnet18-5c106cde.pth", weights_only=False, map_location=device), strict=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device).eval()

# === 預處理流程 ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 判斷圖片 ===
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        prob = probs[pred].item()

    if pred == 0:
        label = "Real (真實照片)"
    else:
        label = "Fake (AI 生成)"

    print(f"預測類別: {pred}, 信心: {prob:.3f}, 結果: {label}")
    return label, prob, image