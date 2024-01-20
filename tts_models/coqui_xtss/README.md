# XTSS 2.0 model

## Creating Virtual environment
1. Firstly install cuda and cudnn using the [Nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
2. Firstly create a conda env using the following command:
    ```bash
        conda create -n xtss python=3.10
    ```
3. Now activate your conda env:
    ```bash
        conda activate xtss
    ```
4. Install the requirements from the [requirements.txt](requirements.txt) (Make sure you are in the correct directory before you run this command)
    ```bash
        pip install -r requirements.txt
    ```
5. Install cuda enabled pytorch using the following command (you can find this command here [pytorch](https://pytorch.org/)):
    ```bash
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

Or import the env using the [xtssConda.yaml](xtssConda.yaml) file


## Fine-Tuning a XTSS 2.0 model locally

1. Make sure you have downloaded this [Github repo](https://github.com/tsurumeso/vocal-remover/releases/tag/v5.1.0) with `baseline.pth` file in the models folder
2. Update the [config.json](config.json) file to point to your data and your config of training 
    - `out_path` – The folder where the dataset will be created and where the runs of the model will be saved.
    - `audio_speaker_list` – A dictionary containing speaker names as keys and lists of corresponding audio file paths as values.
    - `eval_percentage` – The percentage of the dataset to be used for evaluation during training.
    - `language` – The language of the audio data. As of now, XTTS-v2 supports 16 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu) and Korean (ko).
    - `epochs` – The number of times the entire dataset is passed through the model during training.
    - `batch_size` – The number of training examples utilized in one iteration.
    - `grad_acum` – The number of batches to accumulate gradients before performing a backward/update pass. It is recommended to make sure that `batch_size` * `grad_acum` is atleast 252 for more efficient training.
    - `max_audio_length` – The maximum duration (in seconds) of an audio clip.
    - `extract_vocals` - True or false for if you want to first remove all the background noise out of the audio
3. Go into the coqui_xtss folder and run [xtssPipeline.py](xtssPipeline.py) using this command:
    ```bash
        python xtssPipeline.py -c config.json
    ```

## Model outputs

Here you can find some outputs of fine-tuned models [examples](examples/)