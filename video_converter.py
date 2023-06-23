import os
from moviepy.editor import *
from pydub import AudioSegment

class VideoConverter:
    def __init__(self, input_file):
        self.input_file = input_file
        self.video = None
        self.audio = None

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"File not found: {self.input_file}")

        self.load_video()

    def load_video(self):
        try:
            self.video = VideoFileClip(self.input_file)
            self.audio = self.video.audio if self.video.audio is not None else None
        except Exception as e:
            raise Exception(f"Error loading video: {e}")

    def convert_video(self, output_file, format):
        video_codecs = {'webm': 'libvpx', 'wmv': 'wmv2', 'mkv': 'libx264', 'mp4': 'libx264', 'avi': 'libxvid', 'mpeg': 'mpeg2video', 'vob': 'mpeg2video', 'flv': 'flv'}

        if format not in video_codecs:
            raise ValueError(f"Unsupported format: {format}")

        try:
            self.video.write_videofile(output_file, codec=video_codecs[format.lower()], threads=4)
            print(f"Video converted to {format} format successfully!")
            return output_file
        except Exception as e:
            raise Exception(f"Error converting video: {e}")

    def convert_audio(self, output_file, format):
        if format not in ['mp3', 'wav', 'ogg', 'flac', 'aac']:
            raise ValueError(f"Unsupported format: {format}")
        
        if self.audio is None:
            raise Exception("No audio stream found in the input file")
        
        try:
            self.audio.write_audiofile(output_file)
            print(f"Audio converted to {format} format successfully!")
            return output_file
        except Exception as e:
            raise Exception(f"Error converting audio: {e}")

    def convert_to_gif(self, output_file):
        try:
            self.video.write_gif(output_file)
            print(f"Video converted to GIF successfully!")
            return output_file
        except Exception as e:
            raise Exception(f"Error converting video to GIF: {e}")

def convert_video(input_file, format):
    try:
        converter = VideoConverter(input_file)
        if format in ['webm', 'wmv', 'mkv', 'mp4', 'avi', 'mpeg', 'vob', 'flv']:
            return converter.convert_video(f"output.{format}", format), "Converted video successfully!"
        elif format in ['mp3', 'wav', 'ogg', 'flac', 'aac']:
            return converter.convert_audio(f"output.{format}", format), "Converted audio successfully!"
        elif format == "gif":
            return converter.convert_to_gif(f"output.{format}"), "Converted to GIF successfully"
        else:
            return None, "Unsupported format!"
    except Exception as e:
        return None, str(e)