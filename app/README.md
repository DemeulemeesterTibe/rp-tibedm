# Demo application

## Setup 
Rename `api.example.env` to `api.env` and enter your OpenAI API key in the .env file.

To acquire models from the API, you first need to train a model. This can be done using the [Gui](../trainPipeline/README.md) or the [Cli](../tts_models/coqui_xtss/README.md).
Once you have trained your model, copy the folder where your model is located and paste it into a folder named `models`.

## Installation of the app

### Installation using Docker
Ensure you have CUDA and cuDNN installed. You can do this using the [Nvidia documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
Then run the following command:
```
docker compose up 
```
After it is done building what can take quite a while then you can surf to [localhost:3000](http://localhost:3000) and you will see the app.

### Installation Local

1. Install [Node.js](https://nodejs.org/en), [Anaconda](https://www.anaconda.com/download), CUDA, and cuDNN following the [Nvidia documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows).

2. Open a Anaconda PowerShell prompt and run the following commands: 
    ```bash
        conda create -n api python=3.9
        conda activate api 
    ```
    then navigate to the api folder from the Anaconda PowerShell prompt
    ```bash
        cd path/to/the/folder/rp-tibedm/api
    ```
    Install the Python packages and PyTorch using these commands:
    ```bash
        pip install -r requirements.txt
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3. Run `main.py` from the api folder using the newly created Anaconda environment.

4. Open a new terminal and navigate to the web folder:
    ```bash
        cd path/to/the/folder/rp-tibedm/web
    ```
    Install all the npm packages, build the app, and run it using these commands:
    ```bash
        npm i
        npm run build
        npm run start
    ```

5. Now, you can go to [http://localhost:3000](http://localhost:3000) to use the app

## User manual

With the app up and running, either through Docker or locally, you can now use it. There are two main pages:

The `Demo` page, where you need to perform the following steps to generate audio files from ChatGPT responses:

1. Select a **model** from the list and wait for it to load.

2. Select a **speaker** from the speaker list.

3. Select the **language** you want the speaker to use.

4. Type your question in the chatbox or use the record voice button to record your query.

The `Speech Synthesis` page, where you can have the model attempt to clone the speaker from a reference. To make this work, follow these steps: 

1. Go to the **Demo** page and select a model

2. Return to the **Speech Synthesis** page and upload a reference audio file. Remeber a better reference is a better output.

3. Enter the text you want the speaker to say.

4. Listen to the result.

There is also the `Model Differences` page, where you can hear audio clips from other models using the same data and some online services.