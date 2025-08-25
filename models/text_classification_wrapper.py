import logging
import mlflow
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class TextClassificationWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """
        This method is called when loading an MLflow model with this flavor.
        It loads the trained model from artifacts and recreates the tokenizer from its name.
        """
        logger.info("Loading TextClassificationWrapper context.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer configuration from artifacts
        config_path = context.artifacts["wrapper_config"]
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        tokenizer_name = config.get("tokenizer_name", "distilbert-base-uncased")
        logger.info(f"Re-creating tokenizer from name: {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info("Tokenizer re-created successfully.")

        # Load the trained model weights
        model_path = context.artifacts["model_path"]
        logger.info(f"Loading model from path: {model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        logger.info("Model loaded successfully.")

        logger.info(f"Moving model to device: {self.device}...")
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model moved to device and set to eval mode.")

    def predict(self, context, model_input):
        """
        This method is called for every prediction request.
        It expects a pandas DataFrame with a 'text' column.
        """
        if not isinstance(model_input, pd.DataFrame) or 'text' not in model_input.columns:
            raise ValueError("Input must be a pandas DataFrame with a 'text' column.")

        texts = model_input['text'].tolist()
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        return predictions.cpu().numpy()