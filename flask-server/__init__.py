# Inspired by tutorial at https://flask.palletsprojects.com/en/3.0.x/tutorial/factory/

import os
import torch
import torchvision.models as models
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2 as cv
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import matplotlib.pyplot as plt
import numpy as np
import json

from flask import Flask, request, jsonify

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev'
    )
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    """
    Registers the training, validation, and combined datasets, configures the model, and creates a predictor for the Detectron2 model.

    1. Datasets are registered using COCO format with Detectron2's 'register_coco_instances' utility. 
    2. Model parameters like ROI head's test score threshold, test dataset, number of classes, model weights, and device are set in the configuration (cfg). 
    3. The Config object is then used to instantiate a DefaultPredictor for making predictions with the configured model.
    """
    register_coco_instances("beyond_words_train", {}, "../newspaper-navigator-master/beyond_words_data/train_80_percent.json", "../newspaper-navigator-master/beyond_words_data/images")
    register_coco_instances("beyond_words_val", {}, "../newspaper-navigator-master/beyond_words_data/val_20_percent.json", "../newspaper-navigator-master/beyond_words_data/images")
    register_coco_instances("beyond_words_combined", {}, "../newspaper-navigator-master/beyond_words_data/trainval.json", "../newspaper-navigator-master/beyond_words_data/images")

    cfg = get_cfg()

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    cfg.DATASETS.TEST = ("beyond_words_val", )
    cfg.merge_from_file("/home/jakob/anaconda3/envs/JohannaProbiert/lib/python3.7/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

    cfg.MODEL.WEIGHTS = '/home/jakob/Johanna/Projekte/NewspaperNavigator/NewNaApp/NewNaApp/model_final.pth'
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    """
    Converts the given image to binary format using a greyscale threshold.

    Parameters:
    image : numpy array
        The source image in the BGR color space.

    Returns:
    thresh : numpy array
        The binarized version of the input image.
    """
    def to_bin(image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 125, 255, 0)
        plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
        return thresh

    @app.route('/hello')
    def hello():
        return 'Hello, World!'
    
    """
    Receives an image file from a POST request, applies a segmentation model to the image to find advertisements,
    and returns the bounding boxes, confidence scores, and labels of detected advertisements.

    Returns:
    Response object with JSONified list of dictionaries. Each dictionary contains the 
    coordinates, confidence level, and class label of a bounding box. Also includes 
    'Access-Control-Allow-Origin' in the headers for handling Cross-Origin Resource Sharing (CORS).

    If no file is uploaded, returns a response object with an error message.
    """
    @app.route('/getAdvertisementBoxes', methods=['POST'])
    def get_advertisement_boxes():
        if 'imageFile' not in request.files:
            response = jsonify("No file uploaded")
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        image_file = request.files['imageFile']
        nparr = np.frombuffer(image_file.read(), np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        outputs = predictor(img)
        return_object = []
        for i in range(len(outputs["instances"].pred_boxes)):
            box = {}
            box['coords'] = outputs["instances"].pred_boxes[i].tensor.numpy()[0].tolist()
            box['confidence'] = outputs['instances'].scores[i].item()
            box['label'] = outputs['instances'].pred_classes[i].item()
            return_object.append(box)
        print(return_object)
        response = jsonify(return_object)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    """
    Handles file uploading from POST requests, reads the image file, and saves it to local storage.

    Returns:
    Response object with JSON status code 200 if the image is successfully received and stored. 
    Also includes 'Access-Control-Allow-Origin' in the headers for handling Cross-Origin Resource Sharing (CORS).

    If no file is uploaded, returns a response object with an error message.
    """
    @app.route('/registerImage', methods=['POST'])
    def register_image():
        if 'imageFile' not in request.files:
            response = jsonify("No file uploaded")
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        image_file = request.files['imageFile']
        nparr = np.frombuffer(image_file.read(), np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        cv.imwrite('image_upload/original.png', img)
        response = jsonify(200)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    """
    Receives a POST request with JSON data containing bounding boxes. 
    It reads an image from local storage, segments the image based on the provided boxes, 
    then stores the resulting segments.

    Returns:
    Response object with JSON status code 200 if segmentation is successful.
    Adds 'Access-Control-Allow-Origin' in the headers to handle Cross-Origin Resource Sharing (CORS).
    """
    @app.route('/segment', methods=['POST'])
    def segment():
        data = json.loads(request.data)
        print(data)
        boxes = data['rectangles']
        img = cv.imread('image_upload/original.png')
        i = 0
        page_meta = []
        for box in boxes:
            x = int(box['x'])
            y = int(box['y'])
            width = int(box['width'])
            height = int(box['height'])
            cutout = img[y*2:(y+height)*2, x*2:(x+width)*2]
            cv.imwrite(f'image_upload/cutouts/{i}.png', cutout)
            jsonDict = {
                "File": data["file_name"],
                "Size": [width, height],
                "Box": {
                    "top_left": [x*2, y*2],
                    "top_right": [(x+width)*2, y*2],
                    "bottom_left": [x*2, (y+height)*2],
                    "bottom_right": [(x+width)*2, (y+width)*2],
                    "depth": 0,
                    "height": height,
                    "width": width,
                    "size": [width, height],
                    "contains_ad": True,
                    "Percent_page": width*2*height*2/data["page_size"]*2
                }
            }
            page_meta.append({
                "top_left": [x*2, y*2],
                "top_right": [(x+width)*2, y*2],
                "bottom_left": [x*2, (y+height)*2],
                "bottom_right": [(x+width)*2, (y+width)*2],
                "depth": 0,
                "height": height,
                "width": width,
                "size": [width, height],
                "contains_ad": True
            })
            with open(f'image_upload/cutouts/{i}.json', "w") as file:
                file.write(json.dumps(jsonDict))
            with open(f'image_upload/cutouts/page_meta.json', "w") as file:
                file.write(json.dumps(page_meta))
            i += 1
        response = jsonify(200)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    return app