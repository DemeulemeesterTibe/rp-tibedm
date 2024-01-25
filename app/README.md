# Demo application

## Setup 
Change the name of `api.example.env` to `api.env` and put your Openai api key in the .env file

To get models from the api you first have to train a model this you can do using the [Gui](../trainPipeline/README.md) or the [Cli](../tts_models/coqui_xtss/README.md).
When you have trained your model copy the folder that were you model is present and paste it in a folder called `models`

## Installation of the app

### Installation using Docker
Run the following command:
```
docker compose up 
```
After it is done building what can take quite a while then you can surf to [localhost:3000](http://localhost:3000) and you will see the app.

### Installation Local

1. Install [Nodejs](https://nodejs.org/en), [Anaconda](https://www.anaconda.com/download) and Cuda and CUDNN using the [Nvidia docs](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows) .

2. Open a Anaconda powershell prompt and run the following commands: 
    ```bash
        Conda create -n api python=3.9
        Conda activate api 
    ```
    then navigate to the api folder from the Anaconda powershell prompt
    ```bash
        cd path/to/the/folder/rp-tibedm/api
    ```
    Install the python packages and pythorch using these commands:
    ```bash
        pip install -r requirements.txt
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3. Run main.py from the api folder using the newly created Anaconda environment.

4. Open a new terminal and go to the web folder using the cd command:
    ```bash
        cd path/to/the/folder/rp-tibedm/web
    ```
    Now install all the npm packages build the app and run it using these commands:
    ```bash
        npm i
        npm run build
        npm run start
    ```

5. Now you can go to [http://localhost:3000](http://localhost:3000) to use the app

### User manual

Now that you have the app up and running through Docker or running local you can now use the app.
There are 2 main pages you can use.

The `Demo` page were have to do the following things to get audio files from chatgpt responses:

1. Select a **model** from the list and wait until it is done loading.

2. Select a **speaker** from the speaker list.

3. Select the **language** you want the speaker to speak

4. Type your question in the chatbox or use the record voice button to record what you are saying.

The `Speech Synthesis` page were you can have the model try to clone the speaker from a reference to make this work you have to do the following steps: 

1. Go to the **Demo** page and select a model

2. Go back to the **Speech Synthesis** page and upload a reference audio file. 

3. Type the text you want the speaker to say.

4. Listen to the result.

There is also another page the `Model Differences` page were you can hear some audio cliips from other models with the same data used and some online services. 