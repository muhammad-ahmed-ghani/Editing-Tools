import gradio as gr
from watermark_remover import convert_video_to_frames, remove_image_watermark, remove_video_watermark
from video_converter import convert_video
from image_converter import convert_image
from image_editing import edit_image
from image_inpainting import inpaint


css = """
    #remove_btn {
        background: linear-gradient(#201d18, #2bbbc3);
        font-weight: bold;
        font-size: 18px;
        color:white;
    }
    #remove_btn:hover {
        background: linear-gradient(#2bbbc3, #201d18);
    }
    #convert_btn {
        background: linear-gradient(#201d18, #2bbbc3);
        font-weight: bold;
        font-size: 18px;
        color:white;
    }
    #convert_btn:hover {
        background: linear-gradient(#2bbbc3, #201d18);
    }
    #button {
        background: linear-gradient(#201d18, #2bbbc3);
        font-weight: bold;
        font-size: 18px;
        color:white;
    }
    #button:hover {
        background: linear-gradient(#2bbbc3, #201d18);
    }
    footer {
        display: none !important;
    }
"""

demo = gr.Blocks(css=css, title="Editing Tools")
with demo:
    with gr.Tab("Image Converter"):
        gr.Markdown("""
        # <center>🖼️ Image Converter</center>
        """)
        image_format = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'webp', 'ico']
        with gr.Row():
            with gr.Column():
                input_image = gr.File(label="Upload an Image")
            with gr.Column():
                with gr.Row():
                    image_format = gr.Radio(image_format, label="Select Format", interactive=False)
                with gr.Row():
                    image_convert_btn = gr.Button("Convert Image", interactive=False, elem_id="convert_btn")
        with gr.Row():
            output_image = gr.File(label="Output File", interactive=False)
        image_status = gr.Textbox(label="Status", interactive=False)
        input_image.change(lambda x: gr.Radio.update(interactive=True), inputs=[input_image], outputs=[image_format])
        image_format.change(lambda x: gr.Button.update(interactive=True), None, outputs=[image_convert_btn])
        image_convert_btn.click(convert_image, inputs=[input_image, image_format], outputs=[output_image, image_status])

    with gr.Tab("Image Watermark Remover"):
        gr.Markdown("""
        # <center>🖼️ Image Watermark Remover</center>
        """)
        input_image_watermark = gr.Image(label="Upload an Image", tool="sketch", type="pil", interactive=True)
        image_remove_btn = gr.Button("Remove Watermark", interactive=True, elem_id="remove_btn")
        output_image_clean = gr.Image(label="Output Image", interactive=True)

        image_remove_btn.click(remove_image_watermark, inputs=[input_image_watermark], outputs=[output_image_clean])
    
    with gr.Tab("Image Editing"):
        gr.Markdown("""
        # <center>🖼️ Image Editing</center>
        """)
        input_editing_image = gr.Image(label="Upload an Image", type="pil", interactive=True)
        image_editing_options = gr.Radio(["High Res", "Colorize", "Greyscale", "Remove Background"], label="Select Editing Option", interactive=True, value="High Resolution")
        image_editing_btn = gr.Button("Submit", interactive=True, elem_id="button")
        with gr.Row():
            image_editing_output = gr.Image(label="Output Preview", interactive=False)
            image_editing_file = gr.File(label="Download File", interactive=False)

        image_editing_btn.click(edit_image, inputs=[input_editing_image, image_editing_options], outputs=[image_editing_output, image_editing_file])

    with gr.Tab("Image Inpainting"):
        gr.Markdown("""
        # <center>🖼️ Image Inpainting</center>
        """)
        input_inpainting_image = gr.Image(label="Upload an Image", type="pil", interactive=True, tool="sketch")
        input_inpainting_prompt = gr.Textbox(label="Prompt", interactive=True)
        input_inpainting_btn = gr.Button("Submit", interactive=True, elem_id="button")
        with gr.Row():
            input_inpainting_output = gr.Image(label="Image Preview", interactive=False)
            input_inpainting_file = gr.File(label="Download File", interactive=False)

        input_inpainting_btn.click(inpaint, inputs=[input_inpainting_image, input_inpainting_prompt], outputs=[input_inpainting_output, input_inpainting_file])

    with gr.Tab("Video Converter"):
        gr.Markdown("""
        # <center>🎥 Video Converter</center>
        """)
        video_format = ['webm', 'wmv', 'mkv', 'mp4', 'avi', 'mpeg', 'vob', 'flv']
        audio_format = ['mp3', 'wav', 'ogg', 'flac', 'aac']
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Upload a Video")
            with gr.Column():
                with gr.Row():
                    format_select = gr.Radio(["Video", "Audio"], label="Select Format", default="Video")
                with gr.Row():
                    format = gr.Radio(video_format, label="Select Format", interactive=False)
        with gr.Row():
            with gr.Column():
                pass
            with gr.Column():
                convert_btn = gr.Button("Convert Video", interactive=False, elem_id="convert_btn")
            with gr.Column():
                pass
        with gr.Row():
            output = gr.File(label="Output File", interactive=False)
        status = gr.Textbox(label="Status", interactive=False)
        format_select.change(lambda x: gr.Radio.update(choices=video_format if x == "Video" else audio_format, interactive=True), inputs=[format_select], outputs=[format])
        format.change(lambda x: gr.Button.update(interactive=True), None, outputs=[convert_btn])
        convert_btn.click(convert_video, inputs=[input_video, format], outputs=[output, status])
        
    with gr.Tab("Video Watermark Remover"):
        gr.Markdown("""
        # <center>🎥 Video Watermark Remover</center>
        """)
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Upload a Video")
            with gr.Column():
                mask = gr.Image(label="Create a mask for the image", tool="sketch", type="pil", interactive=False)
        with gr.Row():
            with gr.Column():
                pass
            with gr.Column():
                remove_btn = gr.Button("Remove Watermark", interactive=False, elem_id="remove_btn")
            with gr.Column():
                pass
        
        with gr.Row():
            output_video = gr.File(label="Output Video", interactive=False)
        input_video.change(convert_video_to_frames, inputs=[input_video], outputs=[mask, remove_btn])
        remove_btn.click(remove_video_watermark, inputs=[mask], outputs=[output_video, remove_btn])

demo.launch(show_api=False, share=True)
