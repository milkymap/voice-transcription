import io 
import zmq 

import numpy as np 
import operator as op 
import itertools as it, functools as ft 

import torch as th

from os import getenv, path 
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from libraries.log import logger 
from libraries.strategies import *
from dataschema import ZMQServices
    
class ZMQTranscription:
    def __init__(self, model_name, port, worker_signal, synchronizer_condition, sample_rate=16000):
        self.is_initialized = 0 
        
        self.worker_signal = worker_signal 
        self.synchronizer_condition = synchronizer_condition 

        self.port = port 
        self.device = 'cpu'
        self.sample_rate = sample_rate 

        # TRANSFORMERS_CACHE allows to cache the model
        logger.debug('transcriptor will load the model')
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.predictor = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        logger.success(f'the predictor and the processor were loaded on {self.device}')

        self.ctx = zmq.Context()
        self.router_socket = self.ctx.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.LINGER, 0)
        self.router_socket.bind(f'tcp://*:{self.port}')

        self.is_initialized = 1
        
        self.synchronizer_condition.acquire()
        with self.worker_signal.get_lock():
            self.worker_signal.value = self.worker_signal.value - 1
        self.synchronizer_condition.notify_all()  # notify parent process to launch the server 
        self.synchronizer_condition.release()

        
    def __audiofile2audiovector(self, audio_file_data):
        audio_stream = io.BytesIO(audio_file_data)
        audio_segment = AudioSegment.from_file(audio_stream).set_frame_rate(16000)
        duration = audio_segment.duration_seconds 
        audio_vector = np.array(audio_segment.get_array_of_samples())    
        return duration, audio_vector
    
    def close(self):
        if self.is_initialized == 1:
            self.router_socket.close()
            self.ctx.term()  # terminate the zmq context 
        logger.success('transcriptor service has removed all ressources')

    def run(self):
        keep_routing = True 
        while keep_routing:
            logger.debug('transcription is up and waiting for new request')
            caller, delimeter, audio_bytes = self.router_socket.recv_multipart()
            try:
                duration, audio_vector = self.__audiofile2audiovector(audio_bytes)
                audio_tensor = th.as_tensor(audio_vector).float()
                features = self.processor(audio_tensor, sampling_rate=self.sample_rate, return_tensors="pt", padding="longest")
                input_values = features.input_values.to(self.device)
                attention_mask = features.attention_mask.to(self.device)
                with th.no_grad():  # make prediction 
                    logits = self.predictor(input_values, attention_mask=attention_mask).logits.cpu()
                predicted_ids = th.argmax(logits, dim=-1)
                decoded_response = self.processor.batch_decode(predicted_ids)
                transcripted_contents = decoded_response[0].lower()
                self.router_socket.send_multipart([caller, delimeter], flags=zmq.SNDMORE)
                self.router_socket.send_json({
                    'status': 1,
                    'content': {
                        'duration': duration, 
                        'text': transcripted_contents
                    }
                })
            except Exception as e:
                logger.error(e)
                self.router_socket.send_multipart([caller, delimeter], flags=zmq.SNDMORE)
                self.router_socket.send_json({
                    'status': 0, 
                    'content': 'error during transcription'
                })
        # end keep_routing loop 