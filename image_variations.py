import os
import cv2
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL
from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Blip for Image Captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            torch_dtype=torch.float16).to(device)

# ControlNet for Image Variation Generation based on Canny Edge Detection
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", 
    controlnet=ControlNetModel.from_pretrained(
                "thibaud/controlnet-sd21-canny-diffusers", 
                torch_dtype=torch.float16),
    torch_dtype=torch.float16, 
    revision="fp16",
    vae=AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
            ).to(device)
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

def pre_process_image(image):
  image = np.array(image)
  low_threshold = 100
  high_threshold = 200
  image = cv2.Canny(image, low_threshold, high_threshold)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  return Image.fromarray(image)

def image_variations(image, input_prompt):
    canny_image = pre_process_image(image)
    if input_prompt:
        prompt = input_prompt
    else:
        inputs = processor(image, return_tensors="pt").to(device, torch.float16)
        out = model.generate(**inputs)
        prompt = processor.decode(out[0], skip_special_tokens=True)
        print(f"Blip Captioning: {prompt}")
        
    output_images = pipe(
        [prompt]*4,
        canny_image,
        negative_prompt=["distorted, noisy, lowres, bad anatomy, worst quality, low quality, bad eyes, rough face, unclear face"] * 4,
        num_inference_steps=25,
    ).images

    return output_images, canny_image
