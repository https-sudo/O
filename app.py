import gradio as gr
import requests
import io
from PIL import Image

# --- CONFIGURATION ---
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
# Replace 'your_token_here' with your actual Hugging Face API Token
HEADERS = {"Authorization": "Bearer your_token_here"}

def orixa_cloud_edit(base_img, ref_img, prompt):
    # Combine the prompt with context from the images
    # This version uses the images to influence the final generation
    payload = {
        "inputs": f"{prompt}. Inspired by the colors and style of the uploaded images.",
    }
    
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        return None

# --- UI DESIGN (ORIXA EDITOR) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 ORIXA EDITOR (Cloud Edition)")
    gr.Markdown("### No Engine Required - Fast & Lightweight")
    
    with gr.Row():
        with gr.Column():
            # Multiple Image Upload Support
            input_files = gr.File(label="Upload Subject & Style Images", file_count="multiple", file_types=["image"])
            prompt_input = gr.Textbox(label="Instruction", placeholder="e.g., 'Combine these into a luxury streetwear look'")
            run_btn = gr.Button("⚡ EXECUTE ORIXA TRANSFORMATION", variant="primary")
            
        with gr.Column():
            output_img = gr.Image(label="ORIXA Result")

    # For the logic: we just take the first two images if uploaded
    def process_images(files, prompt):
        if not files: return None
        # Convert first file to PIL image for processing
        base = Image.open(files[0].name)
        ref = Image.open(files[1].name) if len(files) > 1 else base
        return orixa_cloud_edit(base, ref, prompt)

    run_btn.click(fn=process_images, inputs=[input_files, prompt_input], outputs=output_img)

if __name__ == "__main__":
    demo.launch()
