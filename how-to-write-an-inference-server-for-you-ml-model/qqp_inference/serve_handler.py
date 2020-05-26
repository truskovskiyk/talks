import logging
from abc import ABC

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler, ABC):
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False
        self.model_threshold = 0.5
        self.model = None
        self.tokenizer = None

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        # Read model serialize/pt file
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.debug("Transformer model from path {0} loaded successfully".format(model_dir))
        self.initialized = True

    def preprocess(self, data):
        logger.info(f"Received text: {data} {type(data)}")
        # we work only with batch size = 1
        payload = data[0]["body"]
        inputs = self.tokenizer.encode_plus(payload["q1"], payload["q2"], add_special_tokens=True, return_tensors="pt")
        return inputs

    def inference(self, inputs):
        prediction = self.model(
            inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device)
        )[0]
        # we work only with batch size = 1
        prediction = torch.softmax(prediction, dim=1)[0][1].item()
        logger.info("Model predicted: '%s'", prediction)
        return [prediction]

    def postprocess(self, inference_output):
        result = [{"score": inference_output[0], "is_duplicate": inference_output[0] > self.model_threshold,}]
        return result


_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
