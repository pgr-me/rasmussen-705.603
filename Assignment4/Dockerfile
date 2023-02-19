FROM python:3.8
WORKDIR /work
COPY . ./

# Install jq to handle JSON, install Python packages, and make directory for Kaggle creds
RUN apt-get update &&\ 
    apt-get install -y jq &&\
    /usr/local/bin/python -m pip install --upgrade pip &&\
    pip3 install -r requirements.txt &&\
    mkdir -p /root/.kaggle /work/data/

# Copy Kaggle username and key into image
RUN echo '{"username": "KAGGLE_USERNAME","key": "KAGGLE_KEY"}' | jq --arg KAGGLE_USERNAME "$KAGGLE_USERNAME" --arg KAGGLE_KEY "$KAGGLE_KEY"  '.username = $KAGGLE_USERNAME | .key = $KAGGLE_KEY' > $HOME/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json
#RUN kaggle datasets download eswarchandt/amazon-music-reviews -p /work/data 


CMD ["hello_world.py"]
ENTRYPOINT ["python"]
