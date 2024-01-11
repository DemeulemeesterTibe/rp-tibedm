import os
from utils import *
import argparse

def train_model():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--data_folder', type=str, help='Path to the dataset folder')
    parser.add_argument('--output_path', type=str, help='Path to the output data')
    parser.add_argument('--language', type=str, help='Language of the dataset')
    parser.add_argument('--epochs', type=float, help='Number of epochs to train')
    parser.add_argument('--grad_acum', type=float, help='Grad accumulation steps')
    parser.add_argument('--batch_size', type=float, help='Batch size used for training')
    parser.add_argument('--speaker_name', type=str, help='Name of the speaker')
    parser.add_argument('--max_audio_length', type=str, help='Max audio length in seconds')
    args = parser.parse_args()

    data_folder = args.data_folder
    output_path = args.output_path
    language = args.language
    num_epochs = args.epochs
    grad_acumm = args.grad_acum
    batch_size = args.batch_size
    speaker_name = args.speaker_name
    max_audio_length = args.max_audio_length

    # get train and eval csv files
    train_csv = os.path.join(data_folder, "metadata_train.csv")
    eval_csv = os.path.join(data_folder, "metadata_eval.csv")

    # model_path = os.path.join(data_folder, "training_model")

    clear_gpu_cache()
    # check if train and eval csv files exist
    print("max_audio_length\t",max_audio_length)
    try:
        max_audio_length = int(max_audio_length * 22050)
    except Exception as e:
        print("error\t",e)
        max_audio_length = 242550
    print("max_audio_length\t",max_audio_length)
    config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_ref = create_train_model(output_path, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, language, speaker_name, max_audio_length)

    config_path_new = os.path.join(exp_path, "original_config.json")
    vocab_file_new = os.path.join(exp_path, "original_vocab.json")
    os.system(f"cp {config_path} {config_path_new}")
    os.system(f"cp {vocab_file} {vocab_file_new}")

    # copy everything from the exp_path to the output_path
    # os.system(f"cp -r {exp_path}/* {output_path}")

    # ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    print("Done Training!")
    clear_gpu_cache()
    # return config_path, vocab_file, ft_xtts_checkpoint, speaker_ref

if __name__ == "__main__":
    train_model()