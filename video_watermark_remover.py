import glob
import os
import io
import ffmpeg
import requests
from PIL import Image
import shutil
import concurrent.futures
import gradio as gr
import cv2
import re


def process_image(mask_data, image_path):
    image = Image.open(image_path)
    image_data = io.BytesIO()
    image.save(image_data, format=image.format)
    image_data = image_data.getvalue()

    # Prepare form data
    form_data = {
            'ldmSteps': 25,
            'ldmSampler': 'plms',
            'zitsWireframe': True,
            'hdStrategy': 'Original',
            'hdStrategyCropMargin': 196,
            'hdStrategyCropTrigerSize': 1280,
            'hdStrategyResizeLimit': 2048,
            'prompt': '',
            'negativePrompt': '',
            'croperX': -24,
            'croperY': -23,
            'croperHeight': 512,
            'croperWidth': 512,
            'useCroper': False,
            'sdMaskBlur': 5,
            'sdStrength': 0.75,
            'sdSteps': 50,
            'sdGuidanceScale': 7.5,
            'sdSampler': 'pndm',
            'sdSeed': 42,
            'sdMatchHistograms': False,
            'sdScale': 1,
            'cv2Radius': 5,
            'cv2Flag': 'INPAINT_NS',
            'paintByExampleSteps': 50,
            'paintByExampleGuidanceScale': 7.5,
            'paintByExampleSeed': 42,
            'paintByExampleMaskBlur': 5,
            'paintByExampleMatchHistograms': False,
            'sizeLimit': 1024,
        }

    files_data = {
        'image': (os.path.basename(image_path), image_data),
        'mask': ('mask.png', mask_data)
    }

    response = requests.post(f'http://localhost:5000/inpaint', data=form_data, files=files_data)

    if response.headers['Content-Type'] == 'image/jpeg' or response.headers['Content-Type'] == 'image/png':
        output_image_path = os.path.join('output_images', os.path.splitext(os.path.basename(image_path))[0] + '_inpainted' + os.path.splitext(image_path)[1])
        with open(output_image_path, 'wb') as output_image_file:
            output_image_file.write(response.content)
    else:
        print(f"Error processing {image_path}: {response.text}")

def remove_watermark(sketch, images_path='frames', output_path='output_images'):
    if os.path.exists('output_images'):
        shutil.rmtree('output_images')
    os.makedirs('output_images')

    mask_data = io.BytesIO()
    sketch["mask"].save(mask_data, format=sketch["mask"].format)
    mask_data = mask_data.getvalue()

    image_paths = glob.glob(f'{images_path}/*.*')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda image_path: process_image(mask_data, image_path), image_paths)

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