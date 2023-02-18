FROM python:3.8
COPY . ./
RUN /usr/local/bin/python -m pip install --upgrade pip &&\
    pip3 install -r requirements.txt
CMD ["hello_world.py"]
ENTRYPOINT ["python"]
