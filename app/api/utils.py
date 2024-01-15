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
