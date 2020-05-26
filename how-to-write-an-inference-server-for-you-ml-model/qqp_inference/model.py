from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os
import boto3
from typing import Any, Union, Dict, List
import torch
import logging
import sys
from pprint import pprint

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


def download_model(bucket_name: str, model_s3_path: str) -> str:
    if os.path.exists(model_s3_path):
        logger.info("Load model from FS")
        return model_s3_path
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)
    logger.info("Load model from S3")
    for object in bucket.objects.filter(Prefix=model_s3_path):
        if not os.path.exists(os.path.dirname(object.key)):
            os.makedirs(os.path.dirname(object.key))
        if not os.path.exists(object.key):
            bucket.download_file(object.key, object.key)
    return model_s3_path


class PythonPredictor:
    def __init__(self, config: Dict[str, Union[str, int, float]]):
        mode_path = download_model(
            bucket_name=config["bucket_name"],
            model_s3_path=config["model_s3_path"])

        self.model_version = config["model_version"]
        self.model_threshold = config["model_threshold"]

        self.tokenizer = DistilBertTokenizer.from_pretrained(mode_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(mode_path)
        self.model.eval()

    def predict(self, payload: Dict[str, str]) -> JSONType:
        with torch.no_grad():
            text_a = payload["q1"]
            text_b = payload["q2"]

            multi_seg_input = self.tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, return_tensors="pt")
            bert_outputs = self.model(input_ids=multi_seg_input["input_ids"], attention_mask=multi_seg_input["attention_mask"])
            # we work only with batch size = 1
            bert_softmax = torch.softmax(bert_outputs[0], dim=1)[0]
            bert_score = bert_softmax[1].item()

            result = {
                "score": bert_score,
                "is_duplicate": bert_score > self.model_threshold,
                "model_version": self.model_version,
            }
            return result

    @classmethod
    def create_for_demo(cls) -> "PythonPredictor":
        config = {
            "bucket_name": "models-for-demo",
            "model_s3_path": "distilbert-qqp",
            "model_version": 1,
            "model_threshold": 0.5,
        }

        return cls(config=config)


if __name__ == "__main__":
    predictor = PythonPredictor.create_for_demo()

    p1 = predictor.predict(
        payload={
            "q1": "What is the responsibility of SAP ERP key user?",
            "q2": "What is a qualified SAP ERP key user?"}
    )
    pprint(p1)

    p2 = predictor.predict(
        payload={
            "q1": "Which is the best book to study TENSOR for general relativity from basic?",
            "q2": "Which is the best book for tensor calculus?"}
    )
    pprint(p2)
