import gc
import os
import shutil
import tempfile
import pandas
from tqdm import tqdm

import torchaudio
import torch

from faster_whisper import WhisperModel

from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from TTS.tts.datasets import load_tts_samples
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from trainer import Trainer, TrainerArgs

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def clear_gpu_cache():
    """Clears the GPU cache."""
    if torch.cuda.is_available():
        # print("Clearing GPU Cache...")
        torch.cuda.empty_cache()

def get_metadata_total_audio_length(audio_files,speaker_name,target_language,whisper_model,out_path,buffer=0.2):
    """Get the metadata for the audio files and the total audio length in seconds.
    Args:
        audio_files (list): A list of audio paths.
        speaker_name (str): The speaker name.
        target_language (str): The target language to use for the transcription.
        whisper_model (WhisperModel): The whisper model to use for the transcription.
        out_path (str): The path to save the metadata files.
        buffer (float): The buffer to use for the audio segments.
        Returns:
            dict: The metadata dictionary.
            float: The total audio size in seconds.
            """
    audio_size = 0

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    for audio_path in tqdm(audio_files):
        wav, sr = torchaudio.load(audio_path)
        # stereo to mono if needed
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_size += (wav.size(-1) / sr)

        print("Transcribing audio file: {}".format(os.path.basename(audio_path)))
        segments, _ = whisper_model.transcribe(audio_path, word_timestamps=True, language=target_language,)
        segments = list(segments)
        print("Audio file {} transcribed. Found {} segments".format(os.path.basename(audio_path), len(segments)))
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        # added all segments words in a unique list
        words_list = []
        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        # process each word
        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start
                # If it is the first sentence, add buffer or get the begining of the file
                if word_idx == 0:
                    sentence_start = max(sentence_start - buffer, 0)  # Add buffer to the sentence start
                else:
                    # get previous sentence end
                    previous_word_end = words_list[word_idx - 1].end
                    # add buffer or get the silence midle between the previous sentence and the current one
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start)/2)

                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            if word.word[-1] in ["!", ".", "?"]:
                sentence = sentence[1:]
                # Expand number and abbreviations plus normalization
                sentence = multilingual_cleaners(sentence, target_language)
                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))

                audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                # Check for the next word's existence
                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                else:
                    # If don't have more words it means that it is the last sentence then use the audio len as next word start
                    next_word_start = (wav.shape[0] - 1) / sr

                # Average the current word end and next word start
                word_end = min((word.end + next_word_start) / 2, word.end + buffer)
                
                absoulte_path = os.path.join(out_path, audio_file)
                os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
                i += 1
                first_word = True

                audio = wav[int(sr*sentence_start):int(sr*word_end)].unsqueeze(0)
                # if the audio is too short ignore it (i.e < 0.33 seconds)
                if audio.size(-1) >= sr/3:
                    torchaudio.save(absoulte_path,
                        audio,
                        sr
                    )
                else:
                    continue

                metadata["audio_file"].append(audio_file)
                metadata["text"].append(sentence)
                metadata["speaker_name"].append(speaker_name)

    return metadata, audio_size

def format_audio_list(audio_speaker_list,out_path,target_language,eval_percentage=0.15):
    """Format the audio list to the format needed for training and evaluation.
    Args:
        audio_speaker_list (dict): A dictionary with the speaker name as key and a list of audio paths as value.
        out_path (str): The path to save the metadata files.
        target_language (str): The target language to use for the transcription.
        eval_percentage (float): The percentage of audio to use for evaluation.
        Returns:
            str: The path to the training metadata file.
            str: The path to the evaluation metadata file.
            float: The total audio size in seconds.
            """
    audio_total_size = 0

    os.makedirs(out_path, exist_ok=True)

    # check if cuda is available and load the model on GPU if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Whisper Model to {}...".format(device))
    whisper_model = WhisperModel("large-v2", device=device, compute_type="float16")
    print("Whisper Model Loaded.")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    for speaker_name, audio_files in audio_speaker_list.items():
        print("Processing speaker: {}".format(speaker_name))
        metadata_temp , audio_size = get_metadata_total_audio_length(audio_files,speaker_name,
                                                                     target_language,whisper_model,out_path)
        audio_total_size += audio_size
        metadata["audio_file"].extend(metadata_temp["audio_file"])
        metadata["text"].extend(metadata_temp["text"])
        metadata["speaker_name"].extend(metadata_temp["speaker_name"])

        print("Speaker {} processed.".format(speaker_name))

    df = pandas.DataFrame(metadata)
    df = df.sample(frac=1)
    num_val_samples = int(len(df)*eval_percentage)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]
        
    df_train = df_train.sort_values('audio_file')
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.to_csv(train_metadata_path, sep="|", index=False)

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval = df_eval.sort_values('audio_file')
    df_eval.to_csv(eval_metadata_path, sep="|", index=False)

    # deallocate VRAM and RAM
    del whisper_model, df_train, df_eval, df, metadata
    gc.collect()
    clear_gpu_cache()
    print("Total audio size: {}".format(audio_total_size))

    return train_metadata_path, eval_metadata_path, audio_total_size


def create_train_model(output_path, train_csv, eval_csv, num_epochs, batch_size, grad_acum, language, speakers ,max_audio_length=255995):
    """
    Creates a new XTTS model and trains it with the given parameters.
    output_path: path to save the model
    train_csv: path to the train csv file
    eval_csv: path to the eval csv file
    num_epochs: number of epochs to train
    batch_size: batch size
    grad_acum: grad accumulation steps
    language: language of the speaker
    speakers: list of speakers
    max_audio_length: max audio length
    Returns: XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_refs
    """
    speaker_name = len(speakers) > 1 and str(len(speakers)) +"_speakers" or str(speakers[0])

    RUN_NAME = speaker_name + "_run"
    PROJECT_NAME = speaker_name + "_project"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    OUT_PATH = os.path.join(output_path, "runs/")
    # OUT_PATH = output_path

    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
    START_WITH_EVAL = False  # if True it will star with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acum  # set here the grad accumulation steps

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name=speaker_name,
        path=os.path.dirname(train_csv),
        meta_file_train=train_csv,
        meta_file_val=eval_csv,
        language=language,
    )
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download the files if they don't exist
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    # XTTS 2.0 checkpoint files
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"


    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))  # config.json file

    # download the files if they don't exist
    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 files!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK, XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )


    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)


    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description=f"""{speaker_name} XTTS training""",
        # dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=12,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=2,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(" > Loaded samples!")
    # print(train_samples)
    # print(eval_samples)

    model = GPTTrainer.init_from_config(config)


    # INITIALIZE THE TRAINER
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ), 
        config, 
        output_path=OUT_PATH, 
        model=model, 
        train_samples=train_samples, 
        eval_samples=eval_samples
    )

    # START TRAINING
    print(" > Starting training!")
    trainer.fit()

    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx =  samples_len.index(max(samples_len))
    # get the speaker reference 
    # speaker_ref = train_samples[longest_text_idx]["audio_file"]
    # get the longest audio file for each speaker
    speaker_refs = {}
    for speaker in speakers:
        speaker_samples = [item for item in train_samples if item["speaker_name"] == speaker]
        samples_len = [len(item["text"].split(" ")) for item in speaker_samples]
        longest_text_idx =  samples_len.index(max(samples_len))
        # speaker_refs.append(speaker_samples[longest_text_idx]["audio_file"])
        speaker_refs[speaker] = speaker_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    del model, trainer, train_samples, eval_samples
    gc.collect()

    return XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_refs

def train_model(language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    """
    Creates a new XTTS model and trains it with the given parameters.
    language: language of the speaker
    train_csv: path to the train csv file
    eval_csv: path to the eval csv file
    num_epochs: number of epochs to train
    batch_size: batch size
    grad_acumm: grad accumulation steps
    output_path: path to save the model
    max_audio_length: max audio length in seconds
    Returns: XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_refs
    """
    # load the train csv and get all the different speakers
    train = pandas.read_csv(train_csv, sep="|")
    speakers = train["speaker_name"].unique()
    
    del train
    gc.collect()
    clear_gpu_cache()
    # check if train and eval csv files exist

    max_audio_length = int(max_audio_length * 22050)

    config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_refs = create_train_model(output_path, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, language, speakers, max_audio_length)

    os.makedirs(os.path.join(exp_path, "speaker_refs"), exist_ok=True)
    
    shutil.copyfile(os.path.join(os.path.dirname(vocab_file), os.path.basename(vocab_file)), os.path.join(exp_path, "vocab.json"))
    shutil.copyfile(os.path.join(os.path.dirname(config_path), os.path.basename(config_path)), os.path.join(exp_path, "original_config.json"))

    for speaker, audio in speaker_refs.items():
        os.makedirs(os.path.join(exp_path, "speaker_refs", speaker), exist_ok=True)
        shutil.copyfile(os.path.join(os.path.dirname(audio), os.path.basename(audio)), os.path.join(exp_path, "speaker_refs", speaker, speaker + ".wav"))

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    # change the name of the checkpoint
    print(" Done Training!")
    clear_gpu_cache()
    return config_path, vocab_file, ft_xtts_checkpoint, speaker_refs

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

def run_tts(model,lang,text,speaker_ref):
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

    if number_of_tokens > 250:
        text_splitting = True

    print(f"Text splitting: {text_splitting}")

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
        enable_text_splitting=text_splitting
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return out_path, speaker_ref