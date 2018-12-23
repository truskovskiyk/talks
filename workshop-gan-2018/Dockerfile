FROM ufoym/deepo:pytorch-py36

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update
RUN apt-get install -y python3.6-tk zlib1g-dev libjpeg-dev

ENV APP_DIR /app
WORKDIR $APP_DIR
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# if CPU SSE4-capable add pillow-simd with AVX2-enabled version
RUN pip uninstall -y pillow
RUN CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

ENV PYTHONPATH $PYTHONPATH:.:/code/
COPY . $APP_DIR