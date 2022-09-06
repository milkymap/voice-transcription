import cv2 
import pickle, json 

import numpy as np 
import pandas as pd 
import operator as op 
import itertools as it, functools as ft 

import nltk 

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F

from os import path, rename 
from glob import glob
from time import time 

from rich.progress import track
from torchvision import transforms as T
from sentence_transformers import SentenceTransformer, util 

from libraries.log import logger 


map_serializer2mode = {json: ('r', 'w'), pickle: ('rb', 'wb')}

def check_env_validity(envar_value, envar_name, envar_type=None):
    if envar_value is None:
        raise ValueError(f'{envar_name} is not defined | please look the helper to see available env variables')
    if envar_type is not None:
        if not op.attrgetter(envar_type)(path)(envar_value):
            raise ValueError(f'{envar_name} should be a valid file or dir')

def measure(func):
    @ft.wraps(func)
    def _measure(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            duration = end_ if end_ > 0 else 0
            logger.debug(f"{func.__name__:<20} total execution time: {duration:04d} ms")
    return _measure

def pull_files(path2location, extension='*'):
    file_paths = sorted(glob(path.join(path2location,'**', extension), recursive=True))
    return file_paths 

def rename_images(path2images, extension):
    image_paths = pull_files(path2images, extension)
    nb_images = len(image_paths)
    logger.debug(f'{nb_images} were found')

    idx = 0
    for path_ in track(image_paths, 'image file rename process'):
        topath = path.join(path2images, f'{idx:03d}.jpg')
        rename(path_, topath)
        idx += 1
    logger.success('all files were renamed')

def to_sentences(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [ sent.strip().lower() for sent in sentences ]
    return sentences

def to_chunks(sentences, chunk_size=7):
    chunks = []
    nb_sentences = len(sentences)
    for idx in range(0, nb_sentences, chunk_size):
        chunks.append('\n'.join(sentences[idx:idx+chunk_size]))
    return chunks 

def serialize(path2location, data, serializer=pickle):
    mode = map_serializer2mode.get(serializer, None)
    if mode is None:
        raise ValueError(f'serializer option must be pickle or json')
    with open(path2location, mode=mode[1]) as fp:
        serializer.dump(data, fp)

def deserialize(path2location, serializer=pickle):
    mode = map_serializer2mode.get(serializer, None)
    if mode is None:
        raise ValueError(f'serializer option must be pickle or json')
    with open(path2location, mode=mode[0]) as fp:
        data = serializer.load(fp)
        return data 

def read_image(path2image):
    cv_image = cv2.imread(path2image, cv2.IMREAD_COLOR)
    cv_image = cv2.resize(cv_image, (256, 256))
    return cv_image 

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

def th2cv(th_image):
    red, green, blue = th_image.numpy() # unpack
    return cv2.merge((blue, green, red))

def scoring(fingerprint, fingerprint_matrix):
    scores = fingerprint @ fingerprint_matrix.T 
    X = np.linalg.norm(fingerprint)
    Y = np.linalg.norm(fingerprint_matrix, axis=1)
    W = X * Y 
    weighted_scores = scores / W 
    return weighted_scores

def vectorize(data, vectorizer, device='cpu', to_tensor=True):
    output = vectorizer.encode(data, device=device, convert_to_tensor=to_tensor)
    return output

def top_k(weighted_scores, k=16):
    scores = th.as_tensor(weighted_scores).float()
    _, indices = th.topk(scores, k, largest=True)
    return indices.tolist()

def prepare_image(th_image):
    normalied_th_image = th_image / th.max(th_image)
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(normalied_th_image)

def load_transformer(path2transformer):
    if path.isfile(path2transformer):
        transformer = deserialize(path2transformer)
        logger.success('the transformer was loaded')
    else:
        try:
            _, model_name = path.split(path2transformer)
            transformer = SentenceTransformer(model_name)
            serialize(path2transformer, transformer)
            logger.success(f'the transformer was downloaded and saved at {path2transformer}')
        except Exception as e:
            logger.error(e)
            raise ValueError(f'{path2transformer} is not a valid path')
    return transformer

if __name__ == '__main__':
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    query_embedding = model.encode('')
    passage_embedding = model.encode([
        'London has 9,787,426 inhabitants at the 2011 census',
        'London is known for its finacial district'
    ])
    print("Similarity:", util.dot_score(query_embedding, passage_embedding))
