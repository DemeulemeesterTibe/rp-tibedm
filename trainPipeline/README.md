# Training an XTSS 2.0 model GUI

## Installation of the gui

### Installation docker
Ensure you have CUDA and cuDNN installed. You can do this using the [Nvidia documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
Then run the following command:
```
docker compose up 
```
After it finishes building, which can take quite a while, you can go to [http://localhost:7860](http://localhost:7860) or use the other link provided in the terminal.

### Installation Local

1. [Anaconda](https://www.anaconda.com/download), CUDA and cuDNN using the [Nvidia documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows).

2. Open a Anaconda PowerShell prompt and run the following commands: 
    ```bash
        conda create -n gui python=3.9
        conda activate gui 
    ```
    Then navigate to the trainPipeline folder from the Anaconda PowerShell prompt
    ```bash
        cd path/to/the/folder/rp-tibedm/trainPipeline
    ```
    Install the Python packages and PyTorch using these commands:
    ```bash
        pip install -r requirements.txt
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
3. Run trainPipeline.py from the trainPipeline folder using the newly created Anaconda environment.

4. Now, you can go to [http://localhost:7860](http://localhost:7860) or use the other link provided in the terminal.

## User manual

Now that you have the GUI running through Docker or you have it running locally you can train your own TTS model. 

In the GUI you have the following field you can change 
- `Speaker name`: Enter the name of the speaker
- `Upload Training Data`: Upload an audio file or files of your speaker. Remember, better data leads to a better model.
- `Language`: The language your speaker is speaking
- `Epochs`: The number of times the model goes through the entire dataset. The higher the number the longer it takes to train.
- `Batch size`: The number of training examples utilized in one iteration. If you encounter  **Cuda out of memory** errors, lower this value. 
- `Gradient accumulation`: The number of batches to accumulate gradients before performing a backward/update pass. Ensure that `batch_size` * `Gradient accumulation` is at least 252 for more efficient training.
- `Max audio length`: The maximum  length of the audio for training in seconds. You can leave this default.
- `Eval percentage`: The percentage of samples used for evaluating the model. You can leave this as the default.

After adjusting the values to your preference, click `Start Pipeline` and wait until the training is complete. Then, you can find your trained model in the `runs/your_speaker_name` folder or in `Dockerdata/your_speaker_name` if you ran the GUI with Docker.
