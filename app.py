from flask import Flask, request, jsonify, render_template
import os
import requests
import numpy as np
import tensorflow as tf
from googletrans import Translator
from googleapiclient.discovery import build
from gtts import gTTS
from io import BytesIO
import base64

app = Flask(__name__)

# Define the path to the uploads directory
uploads_dir = 'D:/recipenator/hackomania/uploads'

# Ensure the uploads directory exists
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Function to load the model
def load_model(model_path):
    if os.path.exists(model_path):
        cnn = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return cnn
    else:
        print("Model path does not exist:", model_path)
        return None

# Function to perform image classification
def perform_image_classification(image_path, model, class_labels):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)  # Get index of the max prediction score
    return class_labels[result_index]

# Function to get recipes based on detected ingredients
def get_recipes(ingredients, api_key, dietary_preference=None, number=1):
    url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "query": ",".join(ingredients),
        "apiKey": api_key,
        "diet": dietary_preference,
        "number": number
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            recipe = data['results'][0]  # Get the first recipe only
            recipe_title = recipe['title']
            recipe_id = recipe['id']
            recipe_image = recipe['image']
            return recipe_title, recipe_id, recipe_image
        else:
            return None, None, None
    else:
        return None, None, None

# Function to get recipe instructions
def get_recipe_instructions(recipe_id, api_key):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/analyzedInstructions"
    params = {"apiKey": api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data:
            instructions = [step['step'] for step in data[0]['steps']]
            return instructions
        else:
            return ["No instructions found for this recipe."]
    else:
        return ["Error fetching instructions. Please try again later."]

# Function to get recipe nutrition
def get_recipe_nutrition(recipe_id, api_key):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/nutritionWidget.json"
    params = {"apiKey": api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data:
            nutrition_info = {
                "calories": data['calories'],
                "carbs": data['carbs'],
                "fat": data['fat'],
                "protein": data['protein']
            }
            return nutrition_info
        else:
            return None
    else:
        return None

# Initialize Google Translate
translator = Translator()

def translate_text(text, dest_language='en'):
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# Initialize YouTube API
youtube_api_key = "YOUR_YOUTUBE_API_KEY"  # Replace with your YouTube API key
youtube = build('youtube', 'v3', developerKey=youtube_api_key)

def get_youtube_video_link(query):
    try:
        request = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=1
        )
        response = request.execute()
        if response['items']:
            video_id = response['items'][0]['id']['videoId']
            return f"https://www.youtube.com/watch?v={video_id}"
        return None
    except Exception as e:
        print(f"YouTube API error: {e}")
        return None

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
        return encoded_audio
    except Exception as e:
        print(f"Text-to-Speech error: {e}")
        return None

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    files = request.files.getlist('file')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected file"}), 400

    detected_ingredients = []

    for file in files:
        if file and allowed_file(file.filename):
            image_path = os.path.join(uploads_dir, file.filename)
            file.save(image_path)
            
            try:
                # Perform image classification
                detected_ingredient = perform_image_classification(image_path, model, class_labels)
                detected_ingredients.append(detected_ingredient)
            except Exception as e:
                print(f"Error during image classification: {e}")
                return jsonify({"error": "Error processing image"}), 500

    # After detecting ingredients, retrieve recipes
    if detected_ingredients:
        recipe_title, recipe_id, recipe_image = get_recipes(detected_ingredients, api_key)
        if recipe_title:
            instructions = get_recipe_instructions(recipe_id, api_key)
            nutrition_info = get_recipe_nutrition(recipe_id, api_key)
            
            # Translate instructions
            translated_instructions = [translate_text(step) for step in instructions]
            
            # Get TTS for instructions
            tts_instructions = [text_to_speech(step) for step in translated_instructions]
            
            # Get YouTube video link
            video_link = get_youtube_video_link(recipe_title)
            
            # Return the recipe details directly
            return jsonify({
                "recipe_title": recipe_title,
                "recipe_image": recipe_image,
                "instructions": translated_instructions,
                "tts_instructions": tts_instructions,  # Include TTS data
                "nutrition_info": nutrition_info,
                "youtube_video": video_link
            })
        else:
            return jsonify({"error": "No recipes found"}), 404
    return jsonify({"error": "No ingredients detected"}), 404

@app.route('/get_recipe', methods=['POST'])
def get_recipe():
    ingredients = request.form.get('ingredients')
    dietary_preference = request.form.get('dietary_preference')
    language = request.form.get('language', 'en')  # Get language preference from form

    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400
    
    ingredients_list = [ingredient.strip() for ingredient in ingredients.split(',')]
    recipe_title, recipe_id, recipe_image = get_recipes(ingredients_list, api_key, dietary_preference)
    
    if recipe_title:
        instructions = get_recipe_instructions(recipe_id, api_key)
        nutrition_info = get_recipe_nutrition(recipe_id, api_key)
        
        # Translate instructions
        translated_instructions = [translate_text(step, dest_language=language) for step in instructions]
        
        # Get TTS for instructions
        tts_instructions = [text_to_speech(step, lang=language) for step in translated_instructions]
        
        # Get YouTube video link
        video_link = get_youtube_video_link(recipe_title)
        
        return jsonify({
            "recipe_title": recipe_title,
            "recipe_image": recipe_image,
            "instructions": translated_instructions,
            "tts_instructions": tts_instructions,  # Include TTS data
            "nutrition_info": nutrition_info,
            "youtube_video": video_link
        })
    return jsonify({"error": "No recipes found"}), 404

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Initialize variables
model_path = "D:/recipenator/HACKATHON/trained_model.h5"
model = load_model(model_path)

# Class labels for image classification
class_labels = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 
    'eggplant', 'garlic', 'ginger', 'grapes', 'kiwi', 'lettuce', 'mango', 'onion', 'orange', 'potato', 'pumpkin', 'spinach', 'tomato', 
    'watermelon'
]

api_key = "YOUR_SPOONACULAR_API_KEY"  # Replace with your Spoonacular API key

if __name__ == "__main__":
    app.run(debug=True)
