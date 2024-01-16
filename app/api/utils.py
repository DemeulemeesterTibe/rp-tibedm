import base64
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch

def clear_gpu_cache():
    """Clears the GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def load_model(checkpoint,config_path, vocab_file):
    """
    Loads a XTTS model from the given files.
    config_path: path to the config file
    vocab_file: path to the vocab file
    ft_xtts_checkpoint: path to the fine-tuned checkpoint
    Returns: XTTS model
    """
    clear_gpu_cache()

    config = XttsConfig()
    config.load_json(config_path)

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=checkpoint, vocab_path=vocab_file, use_deepspeed=False)

    if torch.cuda.is_available():
        model.cuda()

    return model

def run_tts(model,lang,text,speaker_inf):
    """
    Runs the given text through the given model.
    model: XTTS model
    lang: language of the speaker
    text: text to synthesize
    Returns: audio
    """

    out = model.inference(
        text=text,
        language=lang,
        gpt_cond_latent=speaker_inf[0],
        speaker_embedding=speaker_inf[1],
        temperature=model.config.temperature, # Add custom parameters here
        length_penalty=model.config.length_penalty,
        repetition_penalty=model.config.repetition_penalty,
        top_k=model.config.top_k,
        top_p=model.config.top_p,
        # enable_text_splitting=True
    )
    # save it as a wav file
    audio_tensor = torch.from_numpy(out["wav"])

    # Convert the torch tensor to cpu memory
    audio_cpu = audio_tensor.cpu()

    # Convert the cpu tensor to a numpy array
    audio = audio_cpu.numpy()

    # Save it as a wav file
    audio = audio.tobytes()
    wav_base64 = base64.b64encode(audio)
    return wav_base64