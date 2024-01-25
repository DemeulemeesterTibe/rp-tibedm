# Training an XTSS 2.0 model GUI

## Installation of the gui

### Installation docker
Make sure you have Cuda and CUDNN installed you can do this using the [Nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
Then run the following command:
```
docker compose up 
```
After it is done building what can take quite a while then you will get an url in the console use that url to go to the website.

### Installation Local

1. [Anaconda](https://www.anaconda.com/download) and Cuda and CUDNN using the [Nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows).

2. Open a Anaconda powershell prompt and run the following commands: 
    ```bash
        Conda create -n gui python=3.9
        Conda activate gui 
    ```
    then navigate to the trainPipeline folder from the Anaconda powershell prompt
    ```bash
        cd path/to/the/folder/rp-tibedm/trainPipeline
    ```
    Install the python packages and pythorch using these commands:
    ```bash
        pip install -r requirements.txt
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
3. Run trainPipeline.py from the trainPipeline folder using the newly created Anaconda environment.

4. Now you can go to [http://localhost:7860](http://localhost:7860) or use the other link provided in the terminal

## User manual

Now that you have the Gui running through Docker or you have it running locally you can train your own TTS model. 

In the Gui you have the following field you can change 
- `Speaker name`: Fill in the name of the speaker
- `Upload Training Data`: Upload a audio file or files of your speaker. Friendly reminder better data = better model
- `Language`: The language of what your speaker is speaking
- `Epochs`: The number of times the model goes through the whole dataset
- `Batch size`: The batch size is used for training. If you get **Cuda out of memory** errors lower this value. 
- `Gradient accumulation`: Make sure the **Gradient accumulation** * **batch size** is atleast 252. This is for more stable training.
- `Max audio length`: The max length of the audio for training in seconds. You can leave this default.
- `Eval percentage`: Percentage of samples that will be used for evaluation of the model. You can leave this default.

After you have changed all the values to your liking you can click `Start Pipeline` and wait untill the training is done. Than you can find your trained model in the folder `runs/your_speaker_name` or in `Dockerdata/your_speaker_name` if you run the gui with Docker.
