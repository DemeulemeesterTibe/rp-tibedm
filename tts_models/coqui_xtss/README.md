# XTSS 2.0 model

## Creating Virtual environment

## Fine-Tuning a XTSS 2.0 model locally

1. Update the [config.json](config.json) file to point to your data and your config of training 
    - `out_path` – The folder where the dataset will be created and where the runs of the model will be saved.
    - `audio_speaker_list` – A dictionary containing speaker names as keys and lists of corresponding audio file paths as values.
    - `eval_percentage` – The percentage of the dataset to be used for evaluation during training.
    - `language` – The language of the audio data.
    - `epochs` – The number of times the entire dataset is passed through the model during training.
    - `batch_size` – The number of training examples utilized in one iteration.
    - `grad_acum` – The number of batches to accumulate gradients before performing a backward/update pass.
    - `max_audio_length` – The maximum duration (in seconds) of an audio clip.
2. Go into the coqui_xtss folder and run [xtssPipeline.py](xtssPipeline.py) using this command:
    ```bash
        python xtssPipeline.py -c config.json
    ```

## Model outputs

Here are some audio examples: 
<audio controls="controls">
  <source type="audio/wav" src="examples/obama-xtss.wav"></source>
</audio>