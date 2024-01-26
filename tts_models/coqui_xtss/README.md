# XTSS 2.0 model

## Creating Virtual environment
1. Install CUDA and cuDNN following the instructions in the [Nvidia documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
2. Create a Conda environment using the following command:
    ```bash
        conda create -n xtss python=3.9
    ```
3. Activate your Conda environment:
    ```bash
        conda activate xtss
    ```
4. Install the requirements from the [requirements.txt](requirements.txt) file (ensure you are in the correct directory before running this command):
    ```bash
        pip install -r requirements.txt
    ```
5. Install CUDA-enabled PyTorch using the following command:
    ```bash
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

Alternatively, import the environment using the [xtssConda.yaml](xtssConda.yaml) file.

## Fine-Tuning a XTSS 2.0 model locally

1. Download this [GitHub repository](https://github.com/tsurumeso/vocal-remover/releases/tag/v5.1.0) and ensure the `baseline.pth` file is in the models folder.
2. Update the [config.json](config.json) file to specify your data and training configuration: 
    - `out_path` – The folder where the dataset will be created and where the model runs will be saved.
    - `audio_speaker_list` – A dictionary containing speaker names as keys and lists of corresponding audio file paths as values.
    - `eval_percentage` – The percentage of the dataset used for evaluation during training.
    - `language` – The language of the audio data. XTTS-v2 currently supports 16 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu) and Korean (ko).
    - `epochs` – The number of times the entire dataset is passed through the model during training.
    - `batch_size` – The number of training examples utilized in one iteration. If you encounter  **Cuda out of memory** errors, lower this value. 
    - `grad_acum` – The number of batches to accumulate gradients before performing a backward/update pass. Ensure that `batch_size` * `grad_acum` is at least 252 for more efficient training.
    - `max_audio_length` – The maximum duration (in seconds) of an audio clip.
    - `extract_vocals` - Set to true to remove background noise from the audio.

3. Navigate to the coqui_xtss folder and run [xtssPipeline.py](xtssPipeline.py) using this command:
    ```bash
        python xtssPipeline.py -c config.json
    ```

4. When it is done training you can find you trained model in the folder [runs](runs/)

## Model outputs

Find some outputs of fine-tuned models in the [examples](examples/) directory