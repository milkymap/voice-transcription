# genimi-ai
genimi-ai is a tool that evaluate speech and question answering using deep learning

# requirements
```
    before running the script please : 
        create a the next directories at the root of the project:
            mkdir models 
            mkdir models/cache 
```

# docker build and run gpu  
```bash
    docker build -t transcriptor:0.0 -f Dockerfile.gpu . 
    docker run --rm --name genimi --tty --interactive --gpus all -v $(pwd)/models:/home/solver/models -p 8000:8000 transcriptor:0.0 --server_port 8000 
```


# docker build and run cpu  
```bash
    docker build -t transcription:0.0 -f Dockerfile.cpu . 
    docker run --rm --name transcriptor --tty --interactive -v $(pwd)/models:/home/solver/models -p 8000:8000 transcription:0.0 --server_port 8000 
```

# structure of the project

this project is based on opensource libraries such as **[pytorch, zeromq, fastapi, numpy]** 
It contains :
* **services**
    * it contains the transcripion service
    * can be extended for the semantic service 
* **dataschema.py**
    * this file contains definition of datatype for the rest api 
* **libraries**
    * contains usefull functions such as : 
    * log handler 
    * tokenization 
    * features extraction 
    * model loading
* **main.py**
    * this is the entrypoint of the program
    ** it will launch the manager and the server 
    * **.gitignore**
* **.dockerignore**
* **Dockerfile.cpu**
* **Dockerfile.gpu**
* **LICENCE**
* **README.md** 
