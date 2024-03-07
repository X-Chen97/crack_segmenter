import mlflow
from code_src.modelwrapper import ModelWrapper


# download the pretrained model weights from 
# https://datahub.duramat.org/dataset/a1417b84-3724-47bc-90b8-b34660e462bb/resource/45da3b55-fa96-471d-a231-07b98ec5dd8e/download/crack_segmentation.zip
# You need to unzip the file to get the .pt weight
# after you unzip this file, the model.pt file is inside unet_oversample_low_final_model_for_paper
weight_path = 'unet_oversample_low_final_model_for_paper/model.pt'

# we can log the model to mlflow
# since we used modelhandler to train the model, we need to use custom model wrapper 
# to make it compatible with mlflow
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="unet_model",
        python_model=ModelWrapper(), # your custom model wrapper
        artifacts={"model_weight": weight_path},
        code_path=["code_src"],  # Include the path to your code_src package, those functions will be recorded in the mlflow
        registered_model_name="crack_seg_dev"  # Name of the registered model
    )

