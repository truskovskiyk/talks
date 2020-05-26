# How to write an inference server


## Presentation
You can find it [here]()

## Setup

```
virtualenv -p python3.7 ./env
source ./env/bin/activate
pip install -r requirements.txt
```


## Model

We are using [DistilBert](https://arxiv.org/abs/1910.01108) with [QQP dataset](https://gluebenchmark.com/tasks)

To get the model
```
wget https://www.dropbox.com/s/p4lzxl62l8mqdbg/distilbert-qqp.zip
unzip distilbert-qqp.zip
```

(Optional) Upload it to your s3
```
aws s3 cp --recursive distilbert-qqp s3://models-for-demo/distilbert-qqp
```

Try it
```
python ./qqp_inference/model.py
```

# Type 1: Not deploy at all

run jupyter
```
jupyter notebook
```

run streamlit
```
streamlit run streamlit_app.py
```


# Type 2: Web server

call the rest api
```
http POST http://0.0.0.0:8080/predict < example.json
```

run flask 
```
python ./qqp_inference/app_flask.py
gunicorn -w 4 --bind 0.0.0.0:8080 qqp_inference.app_flask:app
```

run fast api

```
uvicorn --host 0.0.0.0 --reload --port 8080 --workers 4 qqp_inference.app_fastapi:app 
``` 

run aiohttp

```
python ./qqp_inference/app_aiohttp.py
gunicorn -w 4 --bind 0.0.0.0:8080 qqp_inference.app_aiohttp:app  --worker-class aiohttp.GunicornWebWorker
```

run torch server

```
torch-model-archiver --model-name "distilbert-qqp" --version 1.0 --serialized-file ./distilbert-qqp/pytorch_model.bin --extra-files "./distilbert-qqp/config.json,./distilbert-qqp/vocab.txt,./distilbert-qqp/tokenizer_config.json" -f --handler "./qqp_inference/serve_handler.py"
torchserve --ts-config torchserve.config --start --model-store ./ --models qqp=distilbert-qqp.mar
torchserve --ts-config torchserve.config --stop --model-store ./ --models qqp=distilbert-qqp.mar


http POST http://0.0.0.0:8080/predictions/qqp < example.json
```


# Type 3: Use a platform

(Optional) Install cortex
```
bash -c "$(curl -sS https://raw.githubusercontent.com/cortexlabs/cortex/0.17/get-cli.sh)"
```

(Optional) Set up a cluster
```
cd cortex-example
cortex cluster up --config=cluster.yaml
cortex cluster info
```


(Optional) Deploy the model
```
cd cortex-example
cortex deploy cortex.yaml
cortex get distilbert-qqp
cortex get distilbert-qqp --watch
cortex delete distilbert-qqp
```

Test deploy

```
http POST http://a9d4dc576763811eaa4ca0a0fd1f5bcd-621843303.us-west-2.elb.amazonaws.com/distilbert-qqp < example.json
```