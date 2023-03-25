import glob
import os
from PIL import Image
import shutil
import concurrent.futures
import gradio as gr
import cv2
import re
import numpy as np
import torch
from lama_cleaner.helper import (
    norm_img,
    get_cache_path_by_url,
    load_jit_model,
)
from lama_cleaner.model.base import InpaintModel
from lama_cleaner.schema import Config

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)
LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")

class LaMa(InpaintModel):
    name = "lama"
    pad_mod = 8

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(LAMA_MODEL_URL))

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res

lama_model = LaMa("cuda" if torch.cuda.is_available() else "cpu")
config = Config(hd_strategy_crop_margin=196, ldm_steps=25, hd_strategy='Original', hd_strategy_crop_trigger_size=1280, hd_strategy_resize_limit=2048)

def remove_image_watermark(inputs):
    alpha_channel = None
    image, mask = inputs["image"], inputs["mask"]
    if image.mode == "RGBA":
        image = np.array(image)
        alpha_channel = image[:, :, -1]
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = np.array(image)
    mask = cv2.threshold(np.array(mask.convert("L")), 127, 255, cv2.THRESH_BINARY)[1]
    output = lama_model(image, mask, config)
    output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != output.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(output.shape[1], output.shape[0])
            )
        output = np.concatenate(
            (output, alpha_channel[:, :, np.newaxis]), axis=-1
        )
    output_image_path = os.path.join('output_images', os.path.splitext(os.path.basename(inputs["image"].filename))[0] + '_inpainted' + os.path.splitext(inputs["image"].filename)[1])
    cv2.imwrite(output_image_path, output)
    return output_image_path

def process_image(mask_data, image_path):
    return remove_image_watermark({"image": Image.open(image_path), "mask": mask_data})

def remove_video_watermark(sketch, images_path='frames', output_path='output_images'):
    if os.path.exists('output_images'):
        shutil.rmtree('output_images')
    os.makedirs('output_images')

    image_paths = glob.glob(f'{images_path}/*.*')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda image_path: process_image(sketch["mask"], image_path), image_paths)

    return gr.File.update(value=convert_frames_to_video('output_images'), visible=True), gr.Button.update(value='Done!')

def convert_video_to_frames(video):
    if os.path.exists('input_video.mp4'):
        os.remove('input_video.mp4')
    # save the video to the current directory from temporary file
    with open(video, 'rb') as f:
        with open('input_video.mp4', 'wb') as f2:
            f2.write(f.read())
    # os.system(f"ffmpeg -i {video} input_video.mp4")
    video_path = 'input_video.mp4'

    if os.path.exists('frames'):
        shutil.rmtree('frames')
    os.makedirs('frames')

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite(f"frames/{video_name}_{count}.jpg", image)
        success, image = vidcap.read()
        count += 1

    return gr.Image.update(value=f"{os.getcwd()}/frames/{video_name}_1.jpg", interactive=True), gr.Button.update(interactive=True)

def convert_frames_to_video(frames_path):
    if os.path.exists('output_video.mp4'):
        os.remove('output_video.mp4')

    img_array = []
    filelist = glob.glob(f"{frames_path}/*.jpg")

    # Sort frames by number
    frame_numbers = [int(re.findall(r'\d+', os.path.basename(frame))[0]) for frame in filelist]
    sorted_frames = [frame for _, frame in sorted(zip(frame_numbers, filelist), key=lambda pair: pair[0])]

    for filename in sorted_frames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    return 'output_video.mp4'