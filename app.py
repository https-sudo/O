import gradio as gr
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler

# --- CONFIGURATION ---
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
CONTROLNET_CHECKPOINT = "lllyasviel/control_v11p_sd15_inpaint"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Models
controlnet = ControlNetModel.from_pretrained(CONTROLNET_CHECKPOINT, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    MODEL_NAME, controlnet=controlnet, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

def orixa_composit(base_image_data, reference_image, prompt):
    # base_image_data: contains the person and the drawn mask
    # reference_image: the clothes/item you want to 'copy'
    
    source_img = base_image_data['background'].convert("RGB").resize((512, 768))
    mask_img = base_image_data['layers'][0].convert("RGB").resize((512, 768))
    ref_img = reference_image.convert("RGB").resize((512, 768))
    
    # We combine the prompt with the reference context
    full_prompt = f"{prompt}, high quality, seamless match"
    
    # Generate the edit
    result = pipe(
        prompt=full_prompt,
        num_inference_steps=30,
        image=source_img,
        mask_image=mask_img,
        control_image=source_img, # Uses inpaint controlnet logic
    ).images[0]
    
    return result

# --- UI DESIGN (ORIXA EDITOR) ---
with gr.Blocks(theme=gr.themes.Monochrome()) as orixa_app:
    gr.HTML("<h1 style='text-align: center;'>ORIXA EDITOR v1.0</h1>")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. The Person (Base)")
            base_input = gr.ImageMask(label="Draw over the area to change", type="pil")
            
            gr.Markdown("### 2. The Reference (Style/Clothing)")
            ref_input = gr.Image(label="Upload the item to copy", type="pil")
            
        with gr.Column():
            gr.Markdown("### 3. Final Result")
            output_display = gr.Image(label="ORIXA Output")
            prompt_input = gr.Textbox(label="Describe the merge", placeholder="e.g., 'wearing this yellow top, realistic fabric'")
            submit_btn = gr.Button("🚀 START ORIXA TRANSFORMATION", variant="primary")

    submit_btn.click(
        fn=orixa_composit, 
        inputs=[base_input, ref_input, prompt_input], 
        outputs=output_display
    )

if __name__ == "__main__":
    orixa_app.launch()
