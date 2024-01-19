import gradio as gr
from utils import *

LANGUAGES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Polish': 'pl',
    'Turkish': 'tr',
    'Russian': 'ru',
    'Dutch': 'nl',
    'Czech': 'cs',
    'Arabic': 'ar',
    'Chinese': 'zh-cn',
    'Japanese': 'ja',
    'Hungarian': 'hu',
    'Korean': 'ko',
    'Hindi': 'hi'
}

def startPipeline(uploaded_files, language):
    print(uploaded_files)
    print(language)
    return "Pipeline finished"

with gr.Blocks() as demo:
    with gr.Tab("Training Pipeline"):
        with gr.Row():
            with gr.Column() as col1:

                gr.Label("Training Pipeline")
                uploaded_files = gr.Files(
                    file_count="multiple",
                    label="Upload Training Data",
                    )
                language = gr.Dropdown(
                    choices=[k for k in LANGUAGES.keys()],
                    value=list(LANGUAGES.keys())[0],
                    label="Language of your dataset",
                    interactive=True,
                    )
            with gr.Column() as col2:

                progress_data = gr.Label(label="Progress")
        prompt_compute_btn = gr.Button(value="Start Pipeline")
    with gr.Tab("Inference Pipeline"):
        gr.Label("Inference Pipeline")
    
    prompt_compute_btn.click(
        fn=startPipeline,
        inputs=[uploaded_files, language],
        outputs=[progress_data],
    )
        

demo.launch(server_port=7878)