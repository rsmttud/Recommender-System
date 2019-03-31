FROM continuumio/anaconda3:latest
RUN apt-get -y update 
RUN apt-get -y install nano
RUN apt-get -y install gcc
RUN apt-get -y install python3

COPY ./Recommender-System /Recommender-System
RUN cd Recommender-System
RUN ls -a
RUN cd ..

RUN yes w | pip install --upgrade pip
RUN yes w | pip install -r /Recommender-System/requirements.txt
RUN yes w | pip install git+https://github.com/boudinfl/pke.git

RUN [ "python3", "-c", "import nltk; nltk.download('all')" ]
RUN python3 -m spacy download en_core_web_sm


WORKDIR /Recommender-System
EXPOSE 80
RUN ls -a
ENTRYPOINT ["python3"]
CMD ["app.py"]
