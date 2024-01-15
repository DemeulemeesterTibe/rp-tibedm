from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch

def clear_gpu_cache():
    """Clears the GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def load_model(ft_xtts_checkpoint,config_path, vocab_file):
    """
    Loads a XTTS model from the given files.
    config_path: path to the config file
    vocab_file: path to the vocab file
    ft_xtts_checkpoint: path to the fine-tuned checkpoint
    Returns: XTTS model
    """
    clear_gpu_cache()
    # load the model
    # model = GPTTrainer.load_from_checkpoint(
    #     checkpoint_path=ft_xtts_checkpoint,
    #     config_path=config_path,
    #     vocab_file=vocab_file,
    #     map_location="cpu",
    # )
    # model.eval()

    config = XttsConfig()
    config.load_json(config_path)
    print("config loaded", config)
    model = Xtts.init_from_config(config)
    # print("model loaded", model)
    model.load_checkpoint(config, checkpoint_path=ft_xtts_checkpoint, vocab_path=vocab_file, use_deepspeed=False)
    if torch.cuda.is_available():
        model.cuda()
    print("model loaded")

    return model

def run_tts(model,lang,text,speaker_ref):
    """
    Runs the given text through the given model.
    model: XTTS model
    lang: language of the speaker
    text: text to synthesize
    Returns: audio
    """

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_ref, gpt_cond_len=model.config.gpt_cond_len, max_ref_length=model.config.max_ref_len, sound_norm_refs=model.config.sound_norm_refs)
    
    out = model.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=model.config.temperature, # Add custom parameters here
        length_penalty=model.config.length_penalty,
        repetition_penalty=model.config.repetition_penalty,
        top_k=model.config.top_k,
        top_p=model.config.top_p,
        # enable_text_splitting=True
    )
    # save it as a wav file
    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    return "created wav", out["wav"]