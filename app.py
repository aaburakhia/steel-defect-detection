# =============================================================================
# app.py - Automated Steel Defect Detection
#
# This application uses a U-Net deep learning model to identify and segment
# manufacturing defects on steel sheets. The UI is built with Gradio Blocks
# to provide a professional, report-oriented user experience.
# =============================================================================

import gradio as gr
from PIL import Image
import numpy as np

# --- Placeholder for your actual model and prediction logic ---
# In a real scenario, you would load your trained model here.
# For this example, we will simulate a model's output.

def predict_defects(input_image):
    """
    This function simulates the output of a segmentation model.
    
    Args:
        input_image (PIL.Image): The image uploaded by the user.
    
    Returns:
        tuple: A tuple containing:
            - A list of (mask, label) tuples for the AnnotatedImage.
            - A string in Markdown format for the analysis report.
    """
    # --- THIS IS WHERE YOUR MODEL'S LOGIC WOULD GO ---
    # 1. Preprocess the input_image.
    # 2. Pass it to your U-Net model.
    # 3. Post-process the model's output masks.
    # 4. Identify the types of defects found.
    # For now, we simulate finding two defects.
    
    # Simulate a "Scratch" mask (a simple rectangle)
    width, height = input_image.size
    scratch_mask = np.zeros((height, width, 4), dtype=np.uint8)
    scratch_mask[int(height*0.4):int(height*0.6), int(width*0.1):int(width*0.7), :] = [255, 0, 0, 180] # Red mask

    # Simulate an "Abrasion" mask (another rectangle)
    abrasion_mask = np.zeros((height, width, 4), dtype=np.uint8)
    abrasion_mask[int(height*0.2):int(height*0.8), int(width*0.8):int(width*0.9), :] = [0, 0, 255, 180] # Blue mask

    # --- Create the outputs for the UI ---
    
    # 1. Annotated Image Output
    # The format is a list of tuples, where each is (mask_numpy_array, label_string)
    annotated_image_data = [
        (scratch_mask, "Scratch"),
        (abrasion_mask, "Abrasion"),
    ]
    
    # 2. Analysis Report Output
    # This is a dynamically generated Markdown string.
    defects_found = ["Scratch", "Abrasion"]
    report_text = f"""
    ### **Defect Analysis Report**
    ---
    **Status:** Defects Detected

    **Summary:** A total of **{len(defects_found)}** defects were identified on the provided steel sheet.

    **Detected Defect Types:**
    """
    for defect in defects_found:
        report_text += f"\n- **{defect}**"
        
    return annotated_image_data, report_text

# =============================================================================
# Gradio UI Definition using gr.Blocks
# =============================================================================

with gr.Blocks(theme='soft', title="Automated Steel Defect Detection") as demo:
    
    # --- Header Section ---
    gr.Markdown(
        """
        # 🧪 Automated Steel Defect Detection
        This application uses a U-Net deep learning model to identify and segment manufacturing defects on steel sheets. 
        This project was developed by a Senior Materials & Project Engineer to bridge deep domain expertise with advanced AI skills.
        """
    )
    
    # --- Input Section ---
    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Steel Sheet Image")
    
    submit_btn = gr.Button("Submit for Analysis", variant="primary")
    
    # --- Output Section ---
    gr.Markdown("--- \n ## 🔎 Analysis Results")
    with gr.Row():
        # Using gr.AnnotatedImage for a more professional and interactive output
        output_image = gr.AnnotatedImage(label="Defect Analysis")
        # Using gr.Markdown for a clean, formatted report
        report_text = gr.Markdown(label="Analysis Report")

    # --- Logic to connect UI components ---
    submit_btn.click(
        fn=predict_defects,
        inputs=input_image,
        outputs=[output_image, report_text]
    )
    
    # --- Example Images ---
    gr.Examples(
        examples=[
            ["./steel_scratch_example.jpg"], # You will need to upload these images to your Space
            ["./steel_rust_example.jpg"],
        ],
        inputs=input_image,
        outputs=[output_image, report_text],
        fn=predict_defects,
        cache_examples=True
    )

# Launch the application
demo.launch()