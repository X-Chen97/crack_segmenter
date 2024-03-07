from pv_vision.nn import ModelHandler
import mlflow
import torch 
from torchvision import transforms
from torch.nn import DataParallel
from code_src.unet_model import construct_unet
from code_src.data_prepare import myDataset
import pandas as pd
import numpy as np

import base64
import cv2 as cv


# this is a custom model wrapper for the pretrained model, which is compatible with mlflow
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        unet = construct_unet(5)
        unet = DataParallel(unet)

        unet.to(self.device)
        unet.load_state_dict(
            torch.load(
                context.artifacts["model_weight"],
                map_location=self.device
                )
            )
        self.model = unet
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            images = []
            for img in model_input['image']:
                img_bytes = base64.b64decode(img)
                # Convert bytes to a numpy array
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                # Decode the numpy array to an image
                img = cv.imdecode(img_array, cv.IMREAD_COLOR)
                images.append(img)
        else:  # Assume input is a list of raw images
            images = model_input
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        imgset = myDataset(images, transform)
        
        self.handler = ModelHandler(
            model=self.model,
            test_dataset=imgset,
            predict_only=True,
            batch_size_val=2,
            device=self.device,
            save_dir='output',
            save_name='unet_prediction'
        )
        masks = self.handler.predict()
        return masks