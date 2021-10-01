# pull base image
FROM alpine:3.5
FROM python:3

VOLUME /var/log/docker

COPY . /tmp/app

COPY SO_vectors_200.bin /tmp/app

WORKDIR /tmp/app/

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

EXPOSE 5001

CMD ["app.py"]