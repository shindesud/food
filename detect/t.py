#python3 train.py --img 640 --batch 16 --epochs 20 --data custom.yaml --weights yolov5s.pt --nosave --cache

import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np
import uuid

app = Flask(__name__)

save_dir = '/root/Food_Calorie_Estimation/detect/images'



@app.route('/' , methods=['POST'])
def get_image():
    file = request.files['image'].read() # Holds byte image
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    print('[INFO] Saving Image..')
    filename = str(uuid.uuid4()) + str('.jpg')
    img_path = os.path.join(save_dir,filename)
    cv2.imwrite(img_path, img)
    print('[INFO] Image Saved!')   

    #Detect function call
    print(img_path)

    return jsonify({'status':'Recieved'})


if __name__ == '__main__':
    #detect(img)
    app.run(debug=True)
