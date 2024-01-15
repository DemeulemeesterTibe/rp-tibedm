import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from utils import *

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/healthcheck')
def healthcheck():
    return {'status': 'Healty'}

@app.get('/')
def default():
    return {'Default': 'Welcome to the XTTS API!'}

@app.get('/getmodels')
def getmodels():
    modelList = os.listdir('models')
    return modelList

@app.get('/selectmodel/{modelname}')
def getmodels():
    modelList = os.listdir('models')
    return modelList

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
