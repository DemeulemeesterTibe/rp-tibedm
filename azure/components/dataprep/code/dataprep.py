
import glob
from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
import gc
from tqdm import tqdm
import torch
import torchaudio
import pandas
from faster_whisper import WhisperModel
import os
import argparse

def get_metadata_total_audio_length(audio_files, target_language="en", out_path=None, buffer=0.2, speaker_name="custom_voice"):
    """Get metadata from audio files and save them in a csv file
    Args:
        audio_files (list): list of audio files
        target_language (str, optional): target language. Defaults to "en".
        out_path ([type], optional): output path. Defaults to None.
        buffer (float, optional): buffer added to the start and end of each sentence number between 0 and 1. Defaults to 0.2.
        speaker_name (str, optional): speaker name. Defaults to "custom_voice".
        Returns:
            metadata (dict): metadata of the audio files
            audio_total_size (float): total audio size in seconds
            """
    audio_total_size = 0
    # make sure that ooutput file exists
    os.makedirs(out_path, exist_ok=True)

    # print lenght of audio files
    print(f"Found {len(audio_files)} audio files!")

    # Loading Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    print(f"Using {device} device")

    print("Loading Whisper Model!")
    asr_model = WhisperModel("large-v2", device=device, compute_type="float32")
    print("Whisper Model Loaded!")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    tqdm_object = tqdm(audio_files)

    for audio_path in tqdm_object:
        wav, sr = torchaudio.load(audio_path)
        # stereo to mono if needed
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        print(f"Transcribing {audio_path}...")
        segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language,)
        segments = list(segments)
        print(f"Found {len(segments)} segments!")
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
    del asr_model
    return metadata, audio_total_size

def main():
    parser = argparse.ArgumentParser(description='Preprocess the dataset')
    parser.add_argument('--data_folder', type=str, help='Path to the dataset folder')
    parser.add_argument('--output_path', type=str, help='Path to the output data')
    parser.add_argument('--language', type=str, help='Language of the dataset')
    parser.add_argument('--buffer', type=float, help='Buffer added to the start and end of each sentence number between 0 and 1')
    parser.add_argument('--eval_percentage', type=float, help='Evaluation percentage number between 0 and 1')
    parser.add_argument('--speaker_name', type=str, help='Name of the speaker')
    args = parser.parse_args()

    data_folder = args.data_folder
    out_path = args.output_path
    target_language = args.language
    buffer = args.buffer / 100
    eval_percentage = args.eval_percentage / 100
    speaker_name = args.speaker_name

    # get all audio files
    audio_files = list(glob.glob(f'{data_folder}/*.mp3') + glob.glob(f'{data_folder}/*.wav'))
    print("audio_files\t",audio_files)
    # audio_files = []
    # for root, _, files in os.walk(data_folder):
    #     for file in files:
    #         if file.endswith(".wav"):
    #             audio_files.append(os.path.join(root, file))
    
    metadata , audio_total_size = get_metadata_total_audio_length(audio_files, target_language, out_path, buffer, speaker_name)

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
    # del df_train, df_eval, df, metadata
    # gc.collect()

    # return train_metadata_path, eval_metadata_path, audio_total_size

if __name__ == "__main__":
    main()