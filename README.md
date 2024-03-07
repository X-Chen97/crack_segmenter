# crack_segmenter
This is a toy system serving the crack segmentation model, as part of [PV-Vision](https://github.com/hackingmaterials/pv-vision) project. Before you start, you need to install the PV-Vision environment.
The codes and models are used for demo only.

## Download the model weights

```bash
curl https://datahub.duramat.org/dataset/a1417b84-3724-47bc-90b8-b34660e462bb/resource/45da3b55-fa96-471d-a231-07b98ec5dd8e/download/crack_segmentation.zip --output model_weights.zip

unzip model_weights.zip
```

## Register the model

You can log the pretrained model and register it to the MLflow server using the following command:
```python
python register_pretrained_model.py
```

Now you should be able to see the model in the MLflow UI. You can start the MLflow server by running the following command:
```bash
mlflow ui
```
Access http://127.0.0.1:5000 in your browser and you will see the registered model:
![alt text](<CleanShot 2024-03-07 at 12.56.17@2x.png>)
Then assign an aliases (e.g., best_model) to this model:
![alt text](<CleanShot 2024-03-07 at 12.58.15@2x.png>)

## serve the registered model

You can serve the model using the following command:
```bash
mlflow models serve -m models:/crack_seg_dev@best_model -p 1234 --env-manager=local
```
I set `--env-manager=local` to use the local environment. Note that my current local environment support pv-vision. You can also use `--env-manager=conda` to setup a new environment.

Now the registered model is serving at http://127.0.0.1:1234

## Get the prediction from the server 

Note that if you just want to run the prediction locally, there is no need to serve the model. You can directly utilize the modelhandler in pv-vision or load the model in mlflow and ust it. Here I just want to show how to get the prediction from the server if you are going to deploy the model on a remote server (e.g., AWS, GCP, etc.).

Run the following command to get the prediction from the server:
```bash
python segment_cracks.py -i img_for_prediction -o output --save_img --url http://127.0.0.1:1234/invocations
```
Replace `img_for_prediction` with your image folder and `output` with your output folder. If you want to save the predicted masks into images, add `--save_img`. The prediction will be saved in the output folder.

Remember to add `invocations` to the end of the url. This is the endpoint for the model serving.

