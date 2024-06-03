from flask import Flask, request, render_template, redirect, url_for, flash
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from androguard.misc import AnalyzeAPK
import tensorflow as tf
import logging

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = 'supersecretkey'

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the models and feature data
model = tf.keras.models.load_model('model/apk_analyzer_model.h5')
cnnmodel = tf.keras.models.load_model('model/apk_analyzer_model_cnn.h5')
feature_df = pd.read_csv("dataset-features-categories.csv", header=None, names=["X", "Category"])

# Extract unique features
permissions_list = feature_df[feature_df["Category"] == "Manifest Permission"].X.unique()
api_call_signatures = feature_df[feature_df["Category"] == "API call signature"].X.unique()
intents = feature_df[feature_df["Category"] == "Intent"].X.unique()
keywords = feature_df[feature_df["Category"] == "Commands signature"].X.unique()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            ann_pred, ann_res, cnn_pred, cnn_res, image_path = analyze_apk(file_path)
            ann_pred_str = '{:.6e}'.format(ann_pred)
            cnn_pred_str = '{:.6e}'.format(cnn_pred)
            return redirect(url_for('result', ann_pred=ann_pred_str, ann_res=ann_res, cnn_pred=cnn_pred_str, cnn_res=cnn_res, image_path=image_path))
    return render_template('index.html')

def analyze_apk(apk_path):
    columns = list(feature_df.X.unique())
    test_df = pd.DataFrame(columns=columns)
    test_df.loc[0] = 0  # Initialize with zeros

    a, d, dx = AnalyzeAPK(apk_path)

    # Extract permissions
    permissions = a.get_permissions()
    found_permissions = [perm.split(".")[-1] for perm in permissions if perm.split(".")[-1] in permissions_list]
    for permission in permissions_list:
        test_df.at[0, permission] = 1 if permission in found_permissions else 0

    # Extract API call signatures
    found_api_signatures = []
    for method in dx.get_methods():
        method_descriptor = method.get_descriptor()
        for api_call in api_call_signatures:
            if re.search(api_call, method_descriptor):
                found_api_signatures.append(api_call)
    for api_call in api_call_signatures:
        test_df.at[0, api_call] = 1 if api_call in found_api_signatures else 0

    # Extract intents
    manifest = a.get_android_manifest_xml()
    intent_filters = manifest.findall(".//intent-filter")
    found_intents = []
    for intent_filter in intent_filters:
        action_elements = intent_filter.findall(".//action")
        for action_element in action_elements:
            action_value = action_element.get("{http://schemas.android.com/apk/res/android}name")
            for intent in intents:
                if re.search(intent, action_value):
                    found_intents.append(intent)
    for intent in intents:
        test_df.at[0, intent] = 1 if intent in found_intents else 0

    # Extract keywords from method code
    found_keywords = []
    for method in dx.get_methods():
        try:
            code = method.get_code().get_instructions()
            for keyword in keywords:
                if re.search(keyword, code):
                    found_keywords.append(keyword)
        except Exception:
            pass
    for keyword in keywords:
        test_df.at[0, keyword] = 1 if keyword in found_keywords else 0

    # Ensure the columns are aligned with the model input
    dropped = test_df.dropna(axis=1, how='all')
    if dropped.shape[1] != 215:
        dropped = dropped.iloc[:, :215]

    # Predict using the loaded models
    ann_pred = model.predict(dropped)[0][0]
    image = convert_to_images(dropped)
    cnn_pred = cnnmodel.predict(image)[0][0]

    ann_res = "malware" if ann_pred > 0.5 else "benign"
    cnn_res = "malware" if cnn_pred > 0.5 else "benign"

    # Save the image
    image_path = save_image(image[0])

    return float(ann_pred), ann_res, float(cnn_pred), cnn_res, image_path

def convert_to_images(data):
    data_array = data.to_numpy()
    required_size = 15 * 15
    padding_size = required_size - data_array.shape[1]
    padded_data = np.pad(data_array, ((0, 0), (0, padding_size)), 'constant', constant_values=0)
    images = padded_data.reshape(-1, 15, 15, 1)  # Adding channel dimension
    return images

def save_image(image_array):
    image_array = image_array.reshape(15, 15)
    plt.imshow(image_array, cmap='gray')
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'apk_image.png')
    plt.axis('off')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    return 'uploads/apk_image.png'  # Return the relative path to static folder

@app.route('/result')
def result():
    ann_pred = request.args.get('ann_pred')
    ann_res = request.args.get('ann_res')
    cnn_pred = request.args.get('cnn_pred')
    cnn_res = request.args.get('cnn_res')
    image_path = request.args.get('image_path')
    return render_template('result.html', ann_pred=ann_pred, ann_res=ann_res, cnn_pred=cnn_pred, cnn_res=cnn_res, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
