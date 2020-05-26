import logging
from abc import ABC

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
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
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.debug("Transformer model from path {0} loaded successfully".format(model_dir))
        self.initialized = True

    def preprocess(self, data):
        logger.info(f"Received text: {data} {type(data)}")
        # we work only with batch size = 1
        payload = data[0]["body"]
        text_a = payload["q1"]
        text_b = payload["q2"]
        multi_seg_input = self.tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, return_tensors="pt")
        return multi_seg_input

    def inference(self, multi_seg_input):
        bert_outputs = self.model(
            input_ids=multi_seg_input["input_ids"], attention_mask=multi_seg_input["attention_mask"]
        )
        # we work only with batch size = 1
        bert_softmax = torch.softmax(bert_outputs[0], dim=1)[0]
        logger.info(f"bert_softmax: {bert_softmax}")
        bert_score = bert_softmax[1].item()
        logger.info(f"Model predicted: '{bert_score}'")
        return [bert_score]

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
