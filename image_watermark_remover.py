import os
import io
import requests
from PIL import Image
NGROK_URL = os.getenv("NGROK_URL")

def remove_image_watermark(input):
    image = input["image"].convert("RGB")
    mask = input["mask"].convert("RGB")
    image_data = io.BytesIO()
    image.save(image_data, format="JPEG")
    image_data = image_data.getvalue()

    mask_data = io.BytesIO()
    mask.save(mask_data, format="JPEG")
    mask_data = mask_data.getvalue()

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
        'image': ('image.jpg', image_data),
        'mask': ('mask.jpg', mask_data),
    }

    response = requests.post(f'{NGROK_URL}/inpaint', data=form_data, files=files_data)

    if response.headers['Content-Type'] == 'image/jpeg' or response.headers['Content-Type'] == 'image/png':
        image = Image.open(io.BytesIO(response.content))
        return image
    else:
        print(f"Error processing Image: {response.text}")