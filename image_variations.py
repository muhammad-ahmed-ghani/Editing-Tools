from PIL import Image
import kornia as K
from kornia.core import Tensor
from diffusers import AutoencoderKL
from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16, 
    revision="fp16",
    vae=AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
            ).to("cuda")
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

def image_variations(filepath):
    img: Tensor = K.io.load_image(filepath, K.io.ImageLoadType.RGB32)    
    img = img[None]
    x_gray = K.color.rgb_to_grayscale(img)     
    x_canny: Tensor = K.filters.canny(x_gray)[0]
    canny_image = K.utils.tensor_to_image(1. - x_canny.clamp(0., 1.0))
    canny_image = Image.fromarray((canny_image * 255).astype('uint8'))
    canny_image = canny_image.convert("L")
    canny_image = canny_image.point(lambda x: 0 if x > 128 else 255, mode="1")
    canny_image = canny_image.convert("RGB")

    inputs = processor(Image.open(filepath), return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs)
    prompt = processor.decode(out[0], skip_special_tokens=True)

    output_images = pipe(
        [prompt]*4,
        canny_image,
        negative_prompt=["distorted, noisy, lowres, bad anatomy, worst quality, low quality, bad eyes"] * 4,
        num_inference_steps=25,
    ).images

    return output_images
