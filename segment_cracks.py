import pandas as pd
import base64
# parse arg
import argparse
from PIL import Image
from imutils.paths import list_images
from torchvision.utils import draw_segmentation_masks
import torch 
from torchvision import transforms
import torchvision.transforms.functional as F
import json
import requests
import numpy as np
import cv2 as cv
from pathlib import Path
import os

def encode_image(x):
    with open(x, "rb") as f:
        x_new = f.read()
    
    return base64.encodebytes(x_new)

def get_masks(masks_raw):
    # masks_raw has the shape batch x n_masks x img_h x img_w, its elements are probability of prediction
    # masks_each has the shape batch x n_masks x img_h x img_w, its elements are boolen (whether belongs to this mask)
    masks_each = []
    # masks_all has the shape batch x img_h x img_w, its elements are labels of masks.
    masks_all = torch.nn.functional.softmax(torch.from_numpy(masks_raw), dim=1).argmax(dim=1)
    for masks in masks_all:
        busbar = masks==1
        crack = masks==2
        cross = masks==3
        dark = masks==4
        masks_each.append(torch.dstack([busbar, crack, cross, dark]).permute(2, 0, 1))
    return masks_each

def draw_mask(img, masks, alpha=0.6):
    img = Image.fromarray(img)
    resize = transforms.Compose([transforms.Resize((256, 256)), transforms.PILToTensor()])
    img = resize(img)

    colors = {
        'dark': (68, 114, 148),
        'cross': (77, 137, 99),
        'crack': (165, 59, 63),
        'busbar': (222, 156, 83)
    }

    combo = draw_segmentation_masks(img, masks, alpha=alpha, colors=[colors[key] for key in ['busbar', 'crack', 'cross', 'dark']])
    return F.to_pil_image(combo)

def run(args):
    # read the image
    image_path = args.image_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_img = args.save_img
    url = args.url
    
    encoded = [encode_image(file) for file in list_images(image_path)]
    images_df = pd.DataFrame(data=encoded, columns=['image'])

    headers = {"Content-Type": "application/json"}
    data = images_df.to_json(orient='split')
    data_js = json.dumps({"dataframe_split": json.loads(data)})
    response = requests.post(url, data=data_js, headers=headers)
    predictions = response.json()

    masks_logit = np.array(predictions['predictions'])
    masks_each = get_masks(masks_logit)

    # save logit and masks to npy
    np.save(Path(output_path)/'masks_logit.npy', masks_logit)
    np.save(Path(output_path)/'masks_each.npy', masks_each)
    
    if save_img:
        images = [cv.imread(file) for file in list_images(image_path)]
        for i, (raw_img, mask) in enumerate(zip(images, masks_each)):
            img = draw_mask(raw_img, mask)
            img.save(Path(output_path)/f'img_{i}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment cracks")
    parser.add_argument("--url", "-u", required=True,
                        type=str, help="URL to the server")
    parser.add_argument("--image_path", "-i", required=True,
                        type=str, help="Path to the images folder")
    parser.add_argument("--output_path", "-o", required=True,
                        type=str, help="Path to the output folder")
    parser.add_argument("--save_img", default=False,
                        action=argparse.BooleanOptionalAction, help="Save the segmented images")
    args = parser.parse_args()
    run(args)
