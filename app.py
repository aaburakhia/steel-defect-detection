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

# --- 2. THE CORE PREDICTION FUNCTION (UPGRADED) ---
# Now returns the image and a dictionary with confidence scores for the label component.
def predict_defects(input_image):
    original_h, original_w, _ = input_image.shape

    transformed = val_transforms(image=input_image)
    image_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_logits = model(image_tensor)

    pred_probs = torch.sigmoid(pred_logits).cpu().numpy().squeeze()
    
    output_image = input_image.copy()
    defect_summary = {}

    for i, class_prob_mask in enumerate(pred_probs):
        # Check if any pixel in the mask has a high probability
        if np.max(class_prob_mask) > 0.5:
            class_id = i + 1
            
            # Get the highest probability value in the mask as the confidence score
            confidence = np.max(class_prob_mask)
            defect_summary[CLASS_NAMES[class_id]] = confidence
            
            # Create the binary mask for drawing
            binary_mask = (class_prob_mask > 0.5).astype(np.uint8)
            resized_mask = cv2.resize(binary_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            
            color = COLORS[i]
            colored_mask = np.zeros_like(output_image)
            colored_mask[resized_mask > 0] = color
            output_image = cv2.addWeighted(output_image, 1, colored_mask, 0.5, 0)

    return output_image, defect_summary

# --- 3. GRADIO INTERFACE ---
with gr.Blocks(theme='gradio/base', css=".footer {display: none !important}") as demo:
    gr.Markdown(
        """
        # Automated Steel Defect Detection
        ### Developed by [Ahmed Aburakhia](https://github.com/aaburakhia) <!-- CHANGED: Personalized byline -->
        <p style='font-size: 16px;'> <!-- CHANGED: Increased font size -->
        This application uses a U-Net deep learning model to identify and segment manufacturing defects on steel sheets. 
        Upload an image or use one of the examples below to see the model in action.
        </p>
        """
    )
    
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Steel Sheet Image")
            submit_button = gr.Button("Submit for Analysis", variant="primary")
        with gr.Column(scale=1):
            image_output = gr.Image(label="Defect Analysis Result")
            label_output = gr.Label(label="Defect Analysis Report") # NEW: More descriptive title

    gr.Markdown("### Click an Example to Start")
    gr.Examples(
        examples=[
            os.path.join("examples", "0b970984e.jpg"),
            os.path.join("examples", "00e0398ad.jpg"),
            os.path.join("examples", "01661826d.jpg")
        ],
        inputs=image_input,
        outputs=[image_output, label_output],
        fn=predict_defects,
        cache_examples=True,
        examples_per_page=3 # CHANGED: Makes examples larger
    )

    with gr.Accordion("View Project Details", open=False): # NEW: Collapsible section for details
        gr.Markdown(
            """
            ### Project Narrative
            As a Senior Materials & Project Engineer with over 12 years in the Saudi Oil & Gas sector, I undertook this project to pivot my career towards R&D and Materials Informatics. The goal was to build an elite-level portfolio piece that directly combines my deep engineering background with advanced AI/ML skills.

            ### Business Value
            Automated defect detection is critical in steel manufacturing for quality control, cost reduction, and safety assurance. This tool can significantly reduce manual inspection time, improve consistency, and provide valuable data for process optimization.

            ### Technical Stack
            - **Model:** U-Net with an EfficientNet-B4 backbone, pre-trained on ImageNet.
            - **Frameworks:** PyTorch, Segmentation Models Pytorch.
            - **Tools:** Kaggle for training, Gradio for the UI, Hugging Face Spaces for deployment.
            - **Key Techniques:** Transfer Learning, Custom Loss Functions (Dice + Focal), Data Augmentation (Albumentations).
            """
        )

    submit_button.click(
        fn=predict_defects,
        inputs=image_input,
        outputs=[image_output, label_output]
    )

if __name__ == "__main__":
    demo.launch()