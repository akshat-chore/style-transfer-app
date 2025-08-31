from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Load model once
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_and_process_image(image_path, target_size=(512, 512)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32) / 255.0
    return img

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/process', methods=['POST'])
def process_style_transfer():
    try:
        content_file = request.files['content']
        style_file = request.files['style']
        intensity = float(request.form.get('intensity', 0.7))
        
        # Save uploaded files
        content_path = os.path.join('uploads', 'content.jpg')
        style_path = os.path.join('uploads', 'style.jpg')
        content_file.save(content_path)
        style_file.save(style_path)
        
        # Process with your ML model
        content_image = load_and_process_image(content_path)
        style_image = load_and_process_image(style_path, (256, 256))
        
        stylized_image = hub_model(content_image, style_image)[0]
        
        # Blend based on intensity
        final_result = intensity * stylized_image + (1 - intensity) * content_image
        
        # Convert to base64
        result_array = (final_result.numpy() * 255).astype(np.uint8)[0]
        result_pil = Image.fromarray(result_array)
        
        buffer = BytesIO()
        result_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'result': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    app.run(debug=True)