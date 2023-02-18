FROM python:3.8
WORKDIR /work
COPY . ./
RUN /usr/local/bin/python -m pip install --upgrade pip &&\
    pip3 install -r requirements.txt &&\
    mkdir -p /root/.kaggle /work/data/ &&\
    echo '{"username": "KAGGLE_USERNAME","key": "KAGGLE_KEY"}' | jq --arg KAGGLE_USERNAME "$KAGGLE_USERNAME" --arg KAGGLE_KEY "$KAGGLE_KEY"  '.username = $KAGGLE_USERNAME | .key = $KAGGLE_KEY' > $HOME/.kaggle/kaggle.json &&\
    kaggle datasets download eswarchandt/amazon-music-reviews -p /work/data 


CMD ["hello_world.py"]
ENTRYPOINT ["python"]
