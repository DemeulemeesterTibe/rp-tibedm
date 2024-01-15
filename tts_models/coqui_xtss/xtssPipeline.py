import argparse
import json
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    
    audio_speaker_list = config['audio_speaker_list']
    out_path = config['out_path']
    dataset_path = os.path.join(out_path, 'dataset')
    language = config['language']
    eval_percentage = config['eval_percentage']
    epochs = config['epochs']
    batch_size = config['batch_size']
    grad_acum = config['grad_acum']
    max_audio_length = config['max_audio_length']

    print("Creating dataset...")
    
    train_metadata_path, eval_metadata_path, audio_total_size = format_audio_list(audio_speaker_list=audio_speaker_list,
                                                                                  out_path=dataset_path,
                                                                                  target_language=language,
                                                                                  eval_percentage=eval_percentage)
    
    print("Done creating dataset")
    print("Training model...")

    configFile, vocabFile, checkpoint, speaker_refs = train_model(language=language,
                                                          train_csv=train_metadata_path,
                                                          eval_csv=eval_metadata_path,
                                                          num_epochs=epochs,
                                                          batch_size=batch_size,
                                                          grad_acumm=grad_acum,
                                                          output_path=out_path,
                                                          max_audio_length=max_audio_length
                                                          )
    print("Done training model")
    print("Model can be found at: ", os.path.dirname(checkpoint))

if __name__ == "__main__":
    main()