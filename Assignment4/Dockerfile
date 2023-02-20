FROM python:3.8
WORKDIR /work

COPY data/Assignment4/. /work/data
COPY Assignment4/. /work

RUN apt-get update &&\ 
    /usr/local/bin/python -m pip install --upgrade pip &&\
    pip3 install -r requirements.txt

CMD [ "python", "import nltk; nltk.download('punkt'); nltk.download('stopwords')" ]


#CMD ["hello_world.py"]
#ENTRYPOINT ["python"]
