from pydantic import BaseModel
import zmq 
import asyncio
import zmq.asyncio 

import cv2 
import numpy as np 

import operator as op 
import itertools as it, functools as ft 

import json 
import pickle 
import hashlib

from fastapi import FastAPI, UploadFile, BackgroundTasks, WebSocket, HTTPException, File
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse 
from fastapi.middleware.cors import CORSMiddleware

from dataschema import ZMQServices
from libraries.log import logger 

app = FastAPI()

origins = ["http://localhost"]
app.add_middleware(
    middleware_class=CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

ctx = None 
server_liveness = None 

async def get_socket_events(socket, socket_poller, time2wait=100):
    incoming_events = dict(await socket_poller.poll(time2wait))  # wait 5s for manager confirmation 
    socket_status = incoming_events.get(socket, None)
    if socket_status is not None:
        if socket_status == zmq.POLLIN:
            return 1
    return 0 

def create_dealer_socket(port):
    global ctx 
    dealer_socket = ctx.socket(zmq.DEALER)
    dealer_socket.setsockopt(zmq.LINGER, 0)
    dealer_socket.connect(f'tcp://localhost:{port}')
    dealer_socket_poller = zmq.asyncio.Poller()
    dealer_socket_poller.register(dealer_socket, zmq.POLLIN)
    return dealer_socket, dealer_socket_poller

@app.on_event('startup')
async def handle_startup():
    global ctx 
    global server_liveness

    ctx = zmq.asyncio.Context()
    server_liveness = asyncio.Event()
    server_liveness.set()  # server is up
    
    logger.success('server is up and ready for getting messages')

@app.on_event('shutdown')
async def handle_shutdown():
    logger.debug('server will go down')
    ctx.term()  # global zeromq context 
    logger.success('server has removed all ressources')

@app.get('/')
async def handle_index():
    return JSONResponse(
        status_code=200, 
        content={
            'message': 'the api is up', 
            'contents': {}
        }
    )

@app.post('/transcript')
async def handle_transcription(incoming_audio: UploadFile=File(...)):
    ZMQ_INIT = 0 
    try:
        audio_bytes = await incoming_audio.read()  # bytestream

        dealer_socket, dealer_socket_poller = create_dealer_socket(ZMQServices.transcription_port)
        ZMQ_INIT = 1 
        await dealer_socket.send_multipart([b'', audio_bytes])
        
        dealer_socket_status = await get_socket_events(dealer_socket, dealer_socket_poller, time2wait=60000)  # wait max 5s
        if dealer_socket_status == 1:  # there is an incoming event 
            if dealer_socket_status == zmq.POLLIN: 
                _, encoded_response = await dealer_socket.recv_multipart()
                decoded_response = json.loads(encoded_response.decode())
                return JSONResponse(
                    status_code=200, 
                    content=decoded_response
                )
        return JSONResponse(
            status_code=400, 
            content={
                'message': 'timeout error : the service take too long time to complete', 
                'content': {}
            }
        )
    except Exception as e:
        exception_value = f'{e}'
        logger.error(e)
        return JSONResponse(
            status_code=400, 
            content={
                'message': exception_value, 
                'content': {}
            }
        )   

    finally:
        if ZMQ_INIT == 1: 
            dealer_socket_poller.unregister(dealer_socket)
            dealer_socket.close()
        logger.debug('prediction was made')    

