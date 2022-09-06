from ast import comprehension
import click 
import uvicorn
import multiprocessing as mp 

from os import getenv, path 

from libraries.log import logger 
from libraries.strategies import check_env_validity

from server import app
from services.transcription import ZMQTranscription
from dataschema import ZMQServices

def start_service(service_name, constructor_, **kwargs):
    try:
        service_ = constructor_(**kwargs)
        service_.run()
    except Exception as e:
        logger.error(e)
    finally:
        logger.success(f'{service_name} service exited')

@click.command()
@click.option('--hostname', default='0.0.0.0')
@click.option('--server_port', type=int, default=8000)  # port of the fastapi server 
@click.option('--transcriptor_name', type=str, default='jonatasgrosman/wav2vec2-large-xlsr-53-french')
def entrypoint(hostname, server_port, transcriptor_name):
    # question_answering if for a next version ...!   
    PROCESSED_LAUNCHED = 0
    try:

        path2models = getenv('MODELS')
        path2transformers_cache = getenv('TRANSFORMERS_CACHE')
        
        check_env_validity(path2models, 'MODELS', 'isdir')
        check_env_validity(path2transformers_cache, 'TRANSFORMERS_CACHE', 'isdir')

        worker_signal = mp.Value('i', 1)  # add the question_answering service 
        services_synchronizer = mp.Condition()

        transcription_process = mp.Process(
            target=start_service,
            args=['transcriptor', ZMQTranscription],
            kwargs={
                'port': ZMQServices.transcription_port, 
                'model_name': transcriptor_name,
                'worker_signal': worker_signal,
                'synchronizer_condition': services_synchronizer
            }
        )

        services_synchronizer.acquire()
        transcription_process.start()
        services_synchronizer.wait_for(lambda: worker_signal.value == 0, timeout=60)

        if worker_signal.value > 0:  
            # timeout was reached and manager was not able to start
            # hence, quit the app 
            logger.error('timeout : workers were not able to start')
            logger.debug('the app will go down!')
            exit(1)

        # app host port 
        server_process = mp.Process(
            target=uvicorn.run, 
            kwargs={
                'app': app, 
                'host': hostname, 
                'port': server_port
            }
        )
        server_process.start()
        
        PROCESSED_LAUNCHED = 1
        server_process.join()
        transcription_process.join()
    except Exception as e:
        logger.error(e)
    finally:
        logger.debug('all ressources (server and workers) were removed')
        if PROCESSED_LAUNCHED == 1:
            server_process.terminate()
            transcription_process.terminate()

if __name__ == '__main__':
    entrypoint()
    
    