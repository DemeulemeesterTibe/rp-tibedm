import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from utils import *
import torch

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

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
def getmodels():
    global MODEL_LIST
    MODEL_LIST = os.listdir('models')
    return MODEL_LIST

@app.get('/get/selected/model')
def selectedModel():
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
def selectmodel():
    global MODEL, MODEL_LIST, MODEL_NAME
    if MODEL is None:
        return {'status': 'No model selected'}
    speaker_dir = os.path.join('models',MODEL_NAME,'speaker_refs')
    return os.listdir(speaker_dir)

@app.get('/select/model/{modelname}')
def selectmodel(modelname):
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
def selectmodel(speakername):
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
    global MODEL, SPEAKER_INF
    if MODEL is None:
        return {'status': 'No model selected'}
    if SPEAKER_INF is None:
        return {'status': 'No speaker selected'}
    print(f"Running TTS for {text}")

    out = MODEL.inference(
        text=text,
        language=lang,
        gpt_cond_latent=SPEAKER_INF[0],
        speaker_embedding=SPEAKER_INF[1],
        temperature=MODEL.config.temperature, # Add custom parameters here
        length_penalty=MODEL.config.length_penalty,
        repetition_penalty=MODEL.config.repetition_penalty,
        top_k=MODEL.config.top_k,
        top_p=MODEL.config.top_p,
        # enable_text_splitting=True
    )
    # save it as a wav file
    audio_tensor = torch.from_numpy(out["wav"])

    # Convert the torch tensor to cpu memory
    audio_cpu = audio_tensor.cpu()

    # Convert the cpu tensor to a numpy array
    audio = audio_cpu.numpy()

    # Save it as a wav file
    audio = audio.tobytes()
    wav_base64 = base64.b64encode(audio)
    print(f"TTS completed for {text}")

    return {'status': 'TTS completed', 'audio': wav_base64.decode()}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
