# Recommender-System

## Installation

Basically,  you have to ways to install the recommendation system: 

1. First you can use the docker image (__recommended__): <a href = "">rs_export.tar</a>
2. Second you can set up a python environment and run the program via _app.py_


### Using the Docker-Container
The docker already holds all necessary files, libraries and dependencies to start the software. Before you start, make 
sure that you have installed docker (https://docs.docker.com/docker-for-windows/install/). 

1. Follow the steps as described here: <a href="">Docker installation</a> to install Docker. 
2. Open the terminal and load the docker image: ```docker load --input [PATH TO DOCKER-FILE] ```
3. Create a docker container with the name "rs" and run it on port 80: ```docker run --name rs  -p 80:80 rs```

#### Some other use full commands
1. Stop a running container: ```Docker stop [NAME OF CONTAINER]``` _(without the square brackets)_
2. Start the docker container again: ```Docker start rs```

### Installation via pip (or Conda)
If you prefer not to use a docker container, you can also install the software to your local python environment. 
We recommend to create a new environment via venv or conda. All required packages are listed in the reqirements.txt file. 
__Please make sure that you use at least Python 3.6.7__

1. First update pip: ``pip install --upgrade pip``
2. Install the packages via pip: `` pip install -r [PATH TO REQUIREMENTS:TXT]`` 
3. After the installation completed run the following command in your python environment to install the pke package: 
    ``pip install git+https://github.com/boudinfl/pke.git``
4. Download all necessary nltk packages: 
    4.1 First type in your terminal (with you corresponding python environment): ``python``
    4.2 Type ``import nltk``
    4.3 Type ``nltk.download("all")``
    4.4 Exit python with: ``exit()``
5. Now you can run the app in you python environment by using following command: ``python [PATH TO app.py]``


## Building a Docker Image
The implemented software may be used for later research projects. If you want to build your own docker image you can use
the docker file attached to the code. But make sure you copy the files outside of the Recommender-System project to build
the container. The directory structure need to be as followed: 

```
- Dockerfile
- .dockerignore
- Recommender-System
    - data
    - models
    - ...
```

1. Open a terminal and navigate yourself to the directory where you copied the Dockerfile: ``cd [PATH TO DOCKERFILE]``
2. Build the image: ```docker build ./ -t rs```
3. Build the container and test it: ``docker run --name rs  -p 80:80 rs``

## Setting up Pycharm 
The whole software engineering process was carried out in PyCharm. Use the following steps if you want to set up the IDE
as we did. 

### Arguments for Pycharm-File-Watcher:
 
 Programm: ```/Library/Ruby/Gems/2.3.0/gems/sass-3.7.2/bin/sass```
 <br/>
 Arguments: ```--no-cache --update $FileName$:../css/$FileNameWithoutExtension$.css```
 <br/>
 Out-Path: ```../css/$FileNameWithoutExtension$.css:../css/$FileNameWithoutExtension$.css.map```
 
### Configuring Flask Configuration

To edit your flask configuration go to: 

1. Run
2. Edit Configurations
3. Add Flask Configuration (Right corner on the top)
4. Change _Target_: ```/PycharmProjects/Recommender-System/app.py```
5. Change _Flask Env_: ```development```
6. Change _Python Interpreter_: ```Project Default..```
7. Activate the Debug mode (Autoreload after changes - only for py files as i noticed.)
8. Run the Application with the flask configuration (On the top left corner where the symbols for run, debug, etc. are)

### Selective Merge using PyCharm
1. Go to Version Control Tab
2. Go to Log
3. Search for the commit you want to merge into your branch and right-click __chery-pick__
4. Commit