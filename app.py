import gradio as gr
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- 1. SETUP: Constants and Model Loading ---
IMG_HEIGHT = 256
IMG_WIDTH = 800
DEVICE = "cpu"

CLASS_NAMES = {
    1: "Pitting / Dots",
    2: "Fine Vertical Lines",
    3: "Scratches / Abrasions",
    4: "Surface Patches"
}
COLORS = [(220, 20, 60), (60, 179, 113), (0, 0, 255), (255, 215, 0)]

print("Defining model architecture...")
model = smp.Unet(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=4,
).to(DEVICE)

MODEL_PATH = "best_model_advanced.pth"
print(f"Loading model from {MODEL_PATH}...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
model.eval()
print("Model loaded successfully.")

val_transforms = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- 2. THE CORE PREDICTION FUNCTION ---
def predict_defects(input_image):
    transformed = val_transforms(image=input_image)
    image_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_logits = model(image_tensor)
    pred_probs = torch.sigmoid(pred_logits)
    pred_masks = (pred_probs > 0.5).cpu().numpy().squeeze()
    output_image = input_image.copy()
    detected_defects = []
    for i, mask in enumerate(pred_masks):
        if mask.sum() > 0:
            class_id = i + 1
            detected_defects.append(CLASS_NAMES[class_id])
            color = COLORS[i]
            colored_mask = np.zeros_like(output_image)
            colored_mask[mask > 0] = color
            output_image = cv2.addWeighted(output_image, 1, colored_mask, 0.5, 0)
    if detected_defects:
        label_text = "Defects Detected: " + ", ".join(detected_defects)
    else:
        label_text = "No Defects Detected"
    cv2.putText(output_image, label_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_image

# --- 3. GRADIO INTERFACE ---
title = "Automated Steel Defect Detection"
description = """
This application uses a U-Net deep learning model to identify and segment manufacturing defects on steel sheets. 
This project was developed by a Senior Materials & Project Engineer to bridge deep domain expertise with advanced AI skills.
<br><br>
**How to use:** Upload an image of a steel sheet, and the model will highlight any detected defects.
"""
iface = gr.Interface(
    fn=predict_defects,
    inputs=gr.Image(type="numpy", label="Upload Steel Sheet Image"),
    outputs=gr.Image(type="numpy", label="Defect Analysis"),
    title=title,
    description=description,
)
if __name__ == "__main__":
    iface.launch()