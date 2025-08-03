import gradio as gr
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

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

# --- 2. THE CORE PREDICTION FUNCTION (REVISED) ---
# This function now returns TWO outputs: the image and a dictionary for the label component.
def predict_defects(input_image):
    original_h, original_w, _ = input_image.shape

    transformed = val_transforms(image=input_image)
    image_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_logits = model(image_tensor)

    pred_probs = torch.sigmoid(pred_logits)
    pred_masks_small = (pred_probs > 0.5).cpu().numpy().squeeze()

    output_image = input_image.copy()
    # This will be the dictionary for the gr.Label component
    defect_summary = {}

    for i, mask_small in enumerate(pred_masks_small):
        if mask_small.sum() > 0:
            class_id = i + 1
            # Add the detected defect to our summary dictionary
            defect_summary[CLASS_NAMES[class_id]] = 1.0
            
            resized_mask = cv2.resize(mask_small.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            
            color = COLORS[i]
            colored_mask = np.zeros_like(output_image)
            colored_mask[resized_mask > 0] = color
            output_image = cv2.addWeighted(output_image, 1, colored_mask, 0.5, 0)

    # If no defects were found, the dictionary will be empty, which is what gr.Label expects.
    return output_image, defect_summary

# --- 3. GRADIO INTERFACE (PROFESSIONAL LAYOUT) ---
# We use gr.Blocks() for full control over the layout.
with gr.Blocks(theme='soft', css=".footer {display: none !important}") as demo:
    gr.Markdown(
        """
        # Automated Steel Defect Detection
        ### Developed by a Senior Materials & Project Engineer
        This application uses a U-Net deep learning model to identify and segment manufacturing defects on steel sheets. 
        Upload an image or use one of the examples below to see the model in action.
        """
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Steel Sheet Image")
            submit_button = gr.Button("Submit", variant="primary")
        with gr.Column():
            image_output = gr.Image(label="Defect Analysis")
            label_output = gr.Label(label="Detected Defect Types")

    gr.Examples(
        examples=[
            os.path.join("examples", "0b970984e.jpg"),
            os.path.join("examples", "00e0398ad.jpg"),
            os.path.join("examples", "01661826d.jpg")
        ],
        inputs=image_input,
        outputs=[image_output, label_output],
        fn=predict_defects,
        cache_examples=True # Speeds up demo for users
    )

    submit_button.click(
        fn=predict_defects,
        inputs=image_input,
        outputs=[image_output, label_output]
    )

if __name__ == "__main__":
    demo.launch()