import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from utils import *
import torch
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


CLIENT_OPENAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = None
MODEL_NAME = None
MODEL_LIST = os.listdir('models')


SPEAKER_INF = None
SPEAKER_NAME = None


@app.get('/healthcheck')
def healthcheck():
    return {'status': 'Healty'}

@app.get('/')
def default():
    return {'Default': 'Welcome to my TTS API'}

@app.get('/get/models')
def getModels():
    global MODEL_LIST
    MODEL_LIST = os.listdir('models')
    return MODEL_LIST

@app.get('/get/selected/model')
def selectedSpeaker():
    global MODEL_NAME
    if MODEL_NAME is None:
        return {'selected': 'None'}
    else:
        return {'selected': MODEL_NAME}

@app.get('/get/selected/speaker')
def selectedModel():
    global SPEAKER_NAME
    if SPEAKER_NAME is None:
        return {'selected': 'None'}
    else:
        return {'selected': SPEAKER_NAME}
    
@app.get('/get/model/speakers')
def getSpeaker():
    global MODEL, MODEL_LIST, MODEL_NAME
    if MODEL is None:
        return {'status': 'No model selected'}
    speaker_dir = os.path.join('models',MODEL_NAME,'speaker_refs')
    return os.listdir(speaker_dir)

@app.get('/select/model/{modelname}')
def selectModel(modelname):
    global MODEL, MODEL_LIST, MODEL_NAME
    if modelname not in MODEL_LIST:
        return {'status': 'Model not found'}
    print(f"Loading model {modelname}")

    config = os.path.join('models',modelname,'config.json')
    vocab = os.path.join('models',modelname,'vocab.json')
    checkpoint = os.path.join('models',modelname,'best_model.pth')

    MODEL = load_model(checkpoint,config,vocab)
    MODEL_NAME = modelname
    print(f"{modelname} model Loaded")

    return {'status': 'Model loaded'}

@app.get('/select/model/speaker/{speakername}')
def selectSpeaker(speakername):
    global MODEL, MODEL_LIST, MODEL_NAME, SPEAKER_INF, SPEAKER_NAME
    print(os.listdir(os.path.join('models',MODEL_NAME,'speaker_refs')))
    if MODEL_NAME is None:
        return {'status': 'No model selected'}
    if speakername not in os.listdir(os.path.join('models',MODEL_NAME,'speaker_refs')):
        return {'status': 'Speaker not found'}
    print(f"Loading speaker {speakername}")

    speaker_ref = os.path.join('models',MODEL_NAME,'speaker_refs',speakername)
    # get the name of the file in speaker_ref
    speaker_ref = os.listdir(speaker_ref)[0]
    speaker_ref = os.path.join('models',MODEL_NAME,'speaker_refs',speakername,speaker_ref)

    gpt_cond_latent, speaker_embedding = MODEL.get_conditioning_latents(audio_path=speaker_ref, gpt_cond_len=MODEL.config.gpt_cond_len, max_ref_length=MODEL.config.max_ref_len, sound_norm_refs=MODEL.config.sound_norm_refs)
    SPEAKER_INF = [gpt_cond_latent, speaker_embedding]
    SPEAKER_NAME = speakername
    print(f"{speakername} speaker Loaded")

    return {'status': 'Speaker loaded'}

@app.post('/run/tts')
def runtts(lang: str, text: str):
    print(lang, text)
    global MODEL, SPEAKER_INF
    if MODEL is None:
        return {'status': 'No model selected'}
    if SPEAKER_INF is None:
        return {'status': 'No speaker selected'}
    print(f"Running TTS for {text}")

    out = run_tts(MODEL, lang, text, SPEAKER_INF)
    print(f"TTS completed for {text}")

    return {'status': 'TTS completed', 'audio': out.decode()}

@app.post('/run/openai/completion')
async def runOpenai(request: Request):
    global CLIENT_OPENAI
    try:
        client_messages = await request.json()
        messages = [
            {"role": "user", "content": message['value']} if message['speaker'] == 'User' else
            {"role": "assistant", "content": message['value']}
            for message in client_messages
        ]
        response = CLIENT_OPENAI.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        text = response.choices[0].message.content

        print(f"OpenAI completion completed for {text}")

        out = run_tts(MODEL, 'en', text, SPEAKER_INF)
        # audio = None

        return {'status': 'OpenAI completion completed', 'text': text, 'audio': out}
    except Exception as e:
        print(f"Error: {e}")
        return {'status': 'Error during OpenAI completion', 'error_message': str(e)}
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
