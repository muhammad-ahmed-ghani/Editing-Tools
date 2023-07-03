import os
import torch
from PIL import Image
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
    image = inputs["image"].resize((image.size[0] - image.size[0] % 64, image.size[1] - image.size[1] % 64), Image.ANTIALIAS)
    mask = inputs["mask"].resize((mask.size[0] - mask.size[0] % 64, mask.size[1] - mask.size[1] % 64), Image.ANTIALIAS)
    # Resize them to have less then 1024 pixels on the largest side by keeping the aspect ratio
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.ANTIALIAS)
    if max(mask.size) > max_size:
        mask.thumbnail((max_size, max_size), Image.ANTIALIAS)
    output = pipe(prompt=prompt, image=image, mask_image=mask, guidance_scale=7.5, height=image.size[1], width=image.size[0])
    output.images[0].save(f"inpainting_output/output.png")
    return output.images[0], "inpainting_output/output.png"
