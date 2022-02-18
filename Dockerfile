FROM tensorflow/tensorflow:2.8.0

WORKDIR /app

RUN python3 -m pip install --upgrade pip \
    && pip3 --disable-pip-version-check --no-cache-dir install \
    pandas scikit-learn pysolar dash dash-bootstrap-components flask_caching gunicorn clearml boto3

COPY . .
RUN python3 setup.py install

CMD gunicorn -b 0.0.0.0:80 app:server
