from flask import Flask, render_template, request
import os
from yolo import process


app = Flask(__name__)


@app.route("/")
def index(name=None):
    return render_template(
        'index.html',
        title="Object Detection Application ",
        description="This website helps us list all objects on an image."
    ) 


@app.route('/upload', methods=['POST'])
def upload_image():
    
    file = request.form.get('file')
    objects, confidence, objects_list = process(file)
    file_name = "New_"+file
    
    return render_template('index.html', file_name=file_name, objects= objects, confidence=confidence, objects_list=objects_list)

if __name__ == "__main__":
    app.run(debug=True)
    