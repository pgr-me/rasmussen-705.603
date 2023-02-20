FROM python:3.8
WORKDIR /work

COPY data/Assignment4/. /work/data
COPY Assignment4/. ./


# Install jq to handle JSON, install Python packages, and make directory for Kaggle creds
RUN apt-get update &&\ 
    /usr/local/bin/python -m pip install --upgrade pip &&\
    pip3 install -r requirements.txt



#CMD ["hello_world.py"]
#ENTRYPOINT ["python"]
