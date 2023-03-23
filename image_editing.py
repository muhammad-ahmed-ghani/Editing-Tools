import os
import sys
from pathlib import Path
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
import numpy as np
import cv2
from PIL import Image
from rembg import remove

# DeOldify
os.system("hub install deoldify==1.2.0")
import paddlehub as hub
hub.server_check()
colorize_model = hub.Module(name='deoldify')

highres_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
bg_upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=highres_model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True
    )

upsampler =  GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    upscale=4,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=bg_upsampler,
    device="cuda" if torch.cuda.is_available() else "cpu",
    )


os.makedirs("deoldify", exist_ok=True)
os.makedirs("gfpganOutput", exist_ok=True)
os.makedirs("greyscale", exist_ok=True)
os.makedirs("rembg", exist_ok=True)

def restore_image(image):
     _, _, output = upsampler.enhance(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), has_aligned=False, only_center_face=False, paste_back=True)
     image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
     return image

def edit_image(image, option):
    tools = ["High Res", "Colorize", "Greyscale", "Remove Background"]
    if option == tools[0]:
        restore_image(image).save("gfpganOutput/output.png")
        return './gfpganOutput/output.png', './gfpganOutput/output.png'
    elif option == tools[1]:
        image.save("deoldify/input.png")
        colorize_model.predict("deoldify/input.png")
        return './output/DeOldify/'+Path('deoldify/input.png').stem+".png", './output/DeOldify/'+Path('deoldify/input.png').stem+".png"
    
    elif option == tools[2]:
        image.convert('L').save("greyscale/output.png")
        return './greyscale/output.png', './greyscale/output.png'
    elif option == tools[3]:
        remove(image).save("rembg/output.png")
        return './rembg/output.png', './rembg/output.png'