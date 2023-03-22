import os
from PIL import Image
import pyheif
import io

class ImageConverter:
    def __init__(self):
        self.supported_formats = {
            'jpg': 'JPEG',
            'jpeg': 'JPEG',
            'png': 'PNG',
            'bmp': 'BMP',
            'tiff': 'TIFF',
            'gif': 'GIF',
            'webp': 'WEBP',
            'ico': 'ICO',
            'heic': 'HEIC',
            'heiv': 'HEIC',
            'heif': 'HEIC',
        }

    def open_heif_image(self, input_image):
        heif_file = pyheif.read(input_image)
        return Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )

    def convert_image(self, input_image, output_format, output_path=None):
        try:
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"The input image '{input_image}' does not exist.")

            input_extension = input_image.split('.')[-1].lower()

            if input_extension not in self.supported_formats:
                raise ValueError(f"The input format '{input_extension}' is not supported.")

            if output_format.lower() not in self.supported_formats:
                raise ValueError(f"The output format '{output_format}' is not supported.")

            if input_extension in ['heic', 'heiv', 'heif']:
                image = self.open_heif_image(input_image)
            else:
                image = Image.open(input_image)

            if output_path is None:
                output_image = '.'.join(input_image.split('.')[:-1]) + f'.{output_format}'
            else:
                output_image = output_path

            image.save(output_image, self.supported_formats[output_format.lower()])
            print(f"Image converted and saved as {output_image}")
            return output_image, "Image converted and saved as {output_image}"
        except Exception as e:
            None, print(f"Error: {e}")

def convert_image(input_image, output_format):
    converter = ImageConverter()
    return converter.convert_image(input_image.name, output_format)