import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import sys

import helper_process_and_predict as dhelper

app = Flask(__name__)

detected_path = 'static/model_output/'
upload_path = 'static/uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

os.system('cd /root/models/research/object_detection/utils')
os.system("sed -i \"s/font = ImageFont.truetype('arial.ttf', 24)/font = ImageFont.truetype('arial.ttf', 50)/\" visualization_utils.py")

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello():
   return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def file_upload():
    global filename
    global input_filename
    global file_type

    #get uploaded file
    if request.method == "POST":
        f = request.files['picfile']
        
        if f.filename.endswith(('.png', '.jpg', '.jpeg')):
            # if file type is correct, save to upload_path
            f.save(os.path.join(upload_path, secure_filename(f.filename)))

            file_type = f.filename.split(".")
            file_type = file_type[1]

            # get output filename
            detected_file = secure_filename(f.filename)
            filename = detected_file
    
            detect_in_image(filename)
            
#            return {
#                    'status_code': 200,
#                    'original_url': 'danms.net' + url_for('static', filename='uploads/' + filename),
#                    'detected_url': 'danms.net' + url_for('static', filename='model_output/' + filename)
                    #}
            return render_template("detection.html", filename=filename)
        return {'status_code': 400, 'message': 'please upload a png, jpg or jpeg.'}
    return {'status_code': 200, 'message': 'GET made.'}

@app.route('/upload_api', methods=['GET', 'POST'])
def file_upload_api():
    global filename
    global input_filename
    global file_type

    #get uploaded file
    if request.method == "POST":
        f = request.files['picfile']
        
        if f.filename.endswith(('.png', '.jpg', '.jpeg')):
            # if file type is correct, save to upload_path
            f.save(os.path.join(upload_path, secure_filename(f.filename)))

            file_type = f.filename.split(".")
            file_type = file_type[1]

            # get output filename
            detected_file = secure_filename(f.filename)
            filename = detected_file
    
            detect_in_image(filename)
            
            return {
                   'status_code': 200,
                   'original_url': 'danms.net' + url_for('static', filename='uploads/' + filename),
                   'detected_url': 'danms.net' + url_for('static', filename='model_output/' + filename)
                    }
        return {'status_code': 400, 'message': 'please upload a png, jpg or jpeg.'}
    return {'status_code': 200, 'message': 'GET made.'}

def detect_in_image(filename):
    sys.path.insert(1, '/root/models/research')
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils

    PATH_TO_SAVED_MODEL="/var/www/another_snakes/saved_model"
    detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
    category_index=label_map_util.create_category_index_from_labelmap("/var/www/another_snakes/label_map.pbtxt",use_display_name=True)
    image_path = "/var/www/another_snakes/static/uploads/" + filename

    image_np = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
        for key, value in detections.items()}

    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.85, # Adjust this value to set the minimum probability boxes to be classified as True
        agnostic_mode=False)

    im = Image.fromarray(image_np_with_detections)
    im.save("/var/www/another_snakes/static/model_output/" + filename)
