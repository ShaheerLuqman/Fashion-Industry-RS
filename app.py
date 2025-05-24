from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image
import io
import os
import pickle
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the pre-trained model
try:
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Load recommendations
try:
    with open('recommendations.pkl', 'rb') as f:
        recommendations = pickle.load(f)
    logger.info("Recommendations loaded successfully")
except Exception as e:
    logger.error(f"Error loading recommendations: {str(e)}")
    raise

def process_image(img):
    try:
        # Resize image to match model's expected sizing
        img = img.resize((224, 224))
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        # Get features
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def find_similar_images(query_features, top_k=6):
    try:
        # Load all image features
        with open('train_features.pkl', 'rb') as f:
            all_features = pickle.load(f)
        
        # Calculate similarities
        similarities = {}
        for img_id, features in all_features.items():
            similarity = np.dot(query_features, features) / (np.linalg.norm(query_features) * np.linalg.norm(features))
            similarities[img_id] = similarity
        
        # Get top k similar images
        similar_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return similar_images
    except Exception as e:
        logger.error(f"Error finding similar images: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    try:
        return send_from_directory('fashion-dataset/images', filename)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {str(e)}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            # Read and process the image
            img = Image.open(io.BytesIO(file.read()))
            features = process_image(img)
            
            # Find similar images
            similar_images = find_similar_images(features)
            
            # Prepare response
            results = []
            for img_id, similarity in similar_images:
                results.append({
                    'image_id': img_id,
                    'similarity': float(similarity),
                    'image_path': f'/images/{img_id}.jpg'
                })
            
            return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 