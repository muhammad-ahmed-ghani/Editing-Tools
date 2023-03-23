import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoencoderKL

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", 
        torch_dtype=torch.float16, 
        revision="fp16",
        vae=AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
            ).to(device)
        ).to(device)
pipe.enable_xformers_memory_efficient_attention()

os.makedirs("inpainting_output", exist_ok=True)

def inpaint(inputs, prompt):
    output = pipe(prompt=prompt, image=inputs["image"], mask_image=inputs["mask"], guidance_scale=7.5)
    output.images[0].save(f"inpainting_output/output.png")
    return output.images[0], "inpainting_output/output.png"
