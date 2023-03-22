import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoencoderKL
from dotenv import load_dotenv
load_dotenv()
ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", 
        torch_dtype=torch.float16, 
        revision="fp16", 
        use_auth_token=ACCESS_TOKEN, 
        vae=AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
            ).to(device)
        ).to(device)

def inpaint(inputs, prompt):
    output = pipe(prompt=prompt, image=inputs["image"], mask_image=inputs["mask"], guidance_scale=7.5)
    return output.images[0]