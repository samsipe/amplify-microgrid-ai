FROM tensorflow/tensorflow:2.8.0

COPY requirements.txt requirements.txt
RUN grep -v "tensor" requirements.txt > tmpfile && mv tmpfile /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/requirements.txt

WORKDIR /app
COPY . .
RUN python3 setup.py install

EXPOSE 80
CMD gunicorn -b 0.0.0.0:80 app:server
