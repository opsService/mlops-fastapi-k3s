import base64
import importlib.util
import io
import logging
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
import cloudpickle
from PIL import Image

logger = logging.getLogger(__name__)


def _load_handler_from_context(context):
    """Helper to load a handler module from artifacts."""
    config_path = context.artifacts["wrapper_config"]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    handler_name = config["handler_name"]
    handler_script_path = Path("/app/models/") / f"{handler_name}.py"
    if not handler_script_path.exists():
        handler_script_path = Path("models/") / f"{handler_name}.py"

    spec = importlib.util.spec_from_file_location(handler_name, handler_script_path)
    handler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handler_module)
    return handler_module, config


class ImageClassificationWrapper(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        """
        This method is called when loading an MLflow model with this flavor.
        It loads the model architecture, weights, and the exact transforms used during training.
        """
        logger.info("Loading ImageClassificationWrapper context.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        handler_module, config = _load_handler_from_context(context)
        self.config = config

        # Re-create model architecture using the handler
        class MockArgs:
            initial_model_file_path = None
            modelName = self.config.get("model_name", "resnet18")

        model = handler_module.create_model(args=MockArgs(), **self.config["data_info"])
        
        # Load weights
        weights_path = context.artifacts["model_weights"]
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model = model.to(self.device)
        self.model.eval()
        logger.info("Model loaded and moved to device.")

        # Load the exact transforms used during training
        transforms_path = context.artifacts["transforms"]
        with open(transforms_path, "rb") as f:
            self.transforms = cloudpickle.load(f)
        logger.info("Transforms loaded successfully.")

    def predict(self, context, model_input):
        """
        This method is called for every prediction request.
        It expects a dictionary with a base64 encoded image string.
        e.g. {"image": "base64..."}
        """
        if not isinstance(model_input, dict) or "image" not in model_input:
            raise ValueError("Input must be a dictionary with an 'image' key.")

        image_b64 = model_input["image"]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        processed_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(processed_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_prob, top_class_idx = torch.topk(probabilities, 1)

        class_names = self.config.get("class_names", [])
        predicted_class_name = class_names[top_class_idx.item()] if len(class_names) > top_class_idx.item() else "unknown_class"
        confidence = top_prob.item()
        
        logger.info(f"Prediction: {predicted_class_name} with confidence {confidence:.4f}")
        return {"prediction": predicted_class_name, "confidence": confidence}