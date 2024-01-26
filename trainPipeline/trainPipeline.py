import gradio as gr
from utils import *
import logging
import sys
import torch
print(torch.cuda.is_available())


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

MODEL = None

class Logger:
    def __init__(self, filename="log.out"):
        self.log_file = filename
        self.terminal = sys.stdout
        self.log = open(self.log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False
    
sys.stdout = Logger()
sys.stderr = sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r") as f:
        return f.read()
    
def load_xtss_model(model_checkpoint, config_path, vocab_path):
    if model_checkpoint == "" or config_path == "" or vocab_path == "":
        return "Please upload all the required files"
    clear_gpu_cache()
    global MODEL
    MODEL = load_model(model_checkpoint, config_path, vocab_path)
    return "Model loaded"

def synthesize(tts_text, speaker_ref, tts_languages):
    if MODEL is None:
        return "Please load a model"
    if tts_text == "" or speaker_ref == "":
        return "Please enter a text and a speaker reference"
    if MODEL is None:
        return "Please load a model"
    tts_language = LANGUAGES[tts_languages]
    tts_audio, ref_audio = run_tts(MODEL, tts_language, tts_text, speaker_ref)
    return "Done synthesize",tts_audio, ref_audio

def startPipeline(uploaded_files, language, epochs, batch_size, grad_acumm, max_audio_length,eval_percentage,speaker_name):
    if uploaded_files is None or len(uploaded_files) == 0:
        return "Please upload at least one file"
    audio_files = [audio_path for audio_path in uploaded_files]
    language = LANGUAGES[language]

    audio_speaker_list = {speaker_name:audio_files}
    out_path = os.path.join(os.getcwd())
    dataset_path = os.path.join(out_path, "dataset")
    print("Creating dataset...")
    train_metadata_path, eval_metadata_path, audio_total_size = format_audio_list(audio_speaker_list=audio_speaker_list,
                                                                                  out_path=dataset_path,
                                                                                  target_language=language,
                                                                                  eval_percentage=eval_percentage)
    
    print("Done creating dataset...")
    print("Training model...")
    configFile, vocabFile, checkpoint, speaker_refs = train_model(language=language,
                                                          train_csv=train_metadata_path,
                                                          eval_csv=eval_metadata_path,
                                                          num_epochs=epochs,
                                                          batch_size=batch_size,
                                                          grad_acumm=grad_acumm,
                                                          output_path=out_path,
                                                          max_audio_length=max_audio_length
                                                          )
    speaker_ref = speaker_refs[speaker_name]
    return "Done training", configFile, vocabFile, checkpoint, speaker_ref
if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Tab("Training Pipeline"):
            with gr.Row():
                with gr.Column() as col1:

                    gr.Label("Training Pipeline")
                    speaker_name = gr.Textbox(
                        label="Speaker name",
                        value="Speaker",
                        )
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
                    epochs = gr.Slider(
                        minimum=1,
                        maximum=500,
                        value=50,
                        step=1,
                        label="Number of epochs",
                        interactive=True,
                        )
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=256,
                        value=8,
                        step=1,
                        label="Batch size",
                        interactive=True,
                        )
                    grad_acumm = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=32,
                        step=1,
                        label="Gradient accumulation. Make sure that batch_size * grad_acumm < 256",
                        interactive=True,
                        )
                    max_audio_length = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=12,
                        step=1,
                        label="Max audio length in seconds",
                        interactive=True,
                        )
                    eval_percentage = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.15,
                        step=0.01,
                        label="Percentage of data used for evaluation",
                        interactive=True,
                        )
                with gr.Column() as col2:
                    logs = gr.Textbox(
                        label="Logs",
                        interactive=False,
                        lines=20,
                    )
                    progress_data = gr.Label(label="Progress")
                    demo.load(read_logs,None,logs,every=1)
            prompt_compute_btn = gr.Button(value="Start Pipeline")
        with gr.Tab("Inference Pipeline"):
            gr.Label("Inference Pipeline")
            with gr.Row():
                with gr.Column() as col1:
                    model_checkpoint = gr.Textbox(
                        label="Model checkpoint",
                        value="",
                        )
                    config_path = gr.Textbox(
                        label="Config file",
                        value="",
                        )
                    vocab_path = gr.Textbox(
                        label="Vocab file",
                        value="",
                        )
                    progress_load = gr.Label(label="Progress")
                    load_btn = gr.Button(value="Load the model")
                with gr.Column() as col2:
                    speaker_ref = gr.Textbox(
                        label="Speaker reference",
                        value="",
                        )
                    tts_languages = gr.Dropdown(
                        choices=[k for k in LANGUAGES.keys()],
                        value=list(LANGUAGES.keys())[0],
                        label="Language of your dataset",
                        interactive=True,
                        )
                    tts_text = gr.Textbox(
                        label="Text to synthesize",
                        value="This is a test sentence",
                        )
                    tts_btn = gr.Button(value="Synthesize")
                with gr.Column() as col3:
                    progress_tts = gr.Label(label="Progress")
                    tts_audio = gr.Audio(label="Synthesized audio")
                    ref_audio = gr.Audio(label="Reference audio")

        
            prompt_compute_btn.click(
                fn=startPipeline,
                inputs=[uploaded_files, language, epochs, batch_size, grad_acumm, max_audio_length,eval_percentage,speaker_name],
                outputs=[progress_data,config_path, vocab_path, model_checkpoint, speaker_ref],
            )
            load_btn.click(
                fn=load_xtss_model,
                inputs=[model_checkpoint, config_path, vocab_path],
                outputs=[progress_load],
            )
            tts_btn.click(
                fn=synthesize,
                inputs=[tts_text, speaker_ref, tts_languages],
                outputs=[progress_tts,tts_audio, ref_audio],
            )
            

    demo.launch(debug=False,share=True,server_name="0.0.0.0")