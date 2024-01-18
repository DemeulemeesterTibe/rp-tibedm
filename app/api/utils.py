import base64
import os
import tempfile
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio

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

    # check if text is more than 250 tokens
    tokenizer = model.tokenizer
    text_splitting = False

    number_of_tokens = len(tokenizer.encode(text,lang))

    print(f"Number of tokens: {number_of_tokens}")

    if number_of_tokens > 250:
        text_splitting = True

    print(f"Text splitting: {text_splitting}")
    
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
        enable_text_splitting=text_splitting
    )

    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)

    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        torchaudio.save(temp_file.name, out["wav"], 24000, bits_per_sample = 16)
        temp_file_path = temp_file.name
    
    # torchaudio.save("test.wav", out["wav"], 24000)

    # Read the contents of the temporary file
    with open(temp_file_path, 'rb') as temp_file:
        file_content = temp_file.read()

    # Encode the contents in base64
    wav_base64 = base64.b64encode(file_content).decode('utf-8')

    # Delete the temporary file
    os.remove(temp_file_path)

    return wav_base64

def synthFromFile(model,lang,text,config,audio_path):
    """
    Runs the given text through the given model.
    model: XTTS model
    lang: language of the speaker
    text: text to synthesize
    Returns: audio
    """

    out = model.synthesize(text=text,config=config,
                       language=lang,speaker_wav=audio_path)
    
    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)

    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        torchaudio.save(temp_file.name, out["wav"], 24000, bits_per_sample = 16)
        temp_file_path = temp_file.name

    # Read the contents of the temporary file
    with open(temp_file_path, 'rb') as temp_file:
        file_content = temp_file.read()

    # Encode the contents in base64
    wav_base64 = base64.b64encode(file_content).decode('utf-8')

    # Delete the temporary file
    os.remove(temp_file_path)
    
    return wav_base64