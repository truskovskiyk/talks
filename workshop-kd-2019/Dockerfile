FROM python:3.7

RUN apt-get update
RUN apt-get install -y htop
ENV APP_DIR /app
WORKDIR $APP_DIR


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN mkdir -p ~/.config/matplotlib/
RUN echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc

ENV PYTHONPATH $PYTHONPATH:.:/app/:
COPY . $APP_DIR

