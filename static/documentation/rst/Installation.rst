Installation of RS
==================


Requirements
------------

- Docker version: 18.09.2
- Python version: 3.6.7

Installation of docker image via batch/shell files (recommended)
----------------------------------------------------------------

1. Follow the steps as described here: `docker installation <https://docs.docker.com/docker-for-windows/install/>`_ to install Docker.
2. Make sure the files listed are in the same directory:

	- *rs_export.tar* 
	- *install.{bat, sh}* 
	- *run.{bat, sh}*
	- *stop{bat, sh}*
3. Run the **install.bat** and wait for the process to finish
4. Go to **localhost:80** in Google Chrome 
5. To stop the docker again run **stop.bat**
6. Steps 3 is only for first installation. In future you simply can run **run.bat**

.. image:: _static/rs.gif

Installation of docker image via command line 
---------------------------------------------

1. Follow the steps as described here: `docker installation <https://docs.docker.com/docker-for-windows/install/>`_ to install Docker.
2. Open Terminal / Command Prompt and load the image via 

.. code-block:: python

   docker load --input [PATH TO rs_export.tar]

3. Create docker container via 

.. code-block:: python

   docker run --name rs -p 80:80 rs 

4. Run the docker via 

.. code-block:: python

	docker run rs


Installation via pip / conda
----------------------------

If you don't prefer to use a docker container, you can also install the software to your local python environment. We recommend to create a new environment via *venv* or *conda*. All required packages are listed in the *requirements.txt* file. Please make sure that you use at least *Python 3.6.7*

1. Update pip

.. code-block:: python

	pip install --upgrade pip

2. Install requirements with: 

.. code-block:: python

	pip install -r [PATH TO requirements.txt]

3. Install *pke-package*:

.. code-block:: python

	pip install git+https://github.com/boudinfl/pke.git

4. Download all nltk packages in your environment from terminal/command prompt:

.. code-block:: python
	
	>>> python
	>>> import nltk
	>>> nltk.download("all")
	>>> exit()

5. Run the application by using: 

.. code-block:: python

	python [PATH TO app.py]

Other useful docker commands
----------------------------

1. Stop a running container:

.. code-block:: python

	docker stop [NAME OF CONTAINER]

2. Start docker container: 

.. code-block:: python

   docker start [NAME OF CONTAINER]

3. List all containers:

.. code-block:: python

   docker ps -a 

4. List all docker images: 

.. code-block:: python

   docker images

5. Delete a docker container:

.. code-block:: python

   docker rm [CONTAINER NAME]

6. Delete a docker image:

.. code-block:: python 

   docker image rm [IMAGE NAME] 


If anything goes wrong please contact:

- Richard Horn (richard.horn94@yahoo.de)
- Daniel Höschele (dhoeschele@outlook.com)