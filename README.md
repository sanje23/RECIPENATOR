# Recipe Finder with Image Classification

This project uses Flask to create a web application that allows users to upload images of ingredients. It uses machine learning (TensorFlow) for image classification to identify the ingredients, then fetches recipes from the Spoonacular API based on the detected ingredients. The app provides recipe instructions, nutrition details, and also converts the instructions to speech using the Google Text-to-Speech API. Additionally, it fetches related YouTube videos for the recipes.

## Features

- **Image Classification:** Uses TensorFlow to classify images of ingredients.
- **Recipe Search:** Retrieves recipes based on the detected ingredients using Spoonacular API.
- **Text Translation:** Translates recipe instructions into the user's preferred language using Google Translate API.
- **Text-to-Speech:** Converts recipe instructions into audio using Google Text-to-Speech API.
- **YouTube Video:** Provides a link to related YouTube videos for the recipe.

## Requirements

- Python 3.8+
- Flask
- TensorFlow
- requests
- googletrans==4.0.0-rc1
- google-api-python-client
- gtts
- dotenv
- werkzeug

## Setup Instructions

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/recipe-finder.git
    cd recipe-finder
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:

      ```bash
      venv\Scripts\activate
      ```

    - On macOS/Linux:

      ```bash
      source venv/bin/activate
      ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Set up your `.env` file with the necessary API keys:

    - `SPOONACULAR_API_KEY`: API key from [Spoonacular](https://spoonacular.com/food-api).
    - `YOUTUBE_API_KEY`: API key from [Google Developers Console](https://console.developers.google.com/).

    Example `.env`:

    ```
    SPOONACULAR_API_KEY=your_spoonacular_api_key
    YOUTUBE_API_KEY=your_youtube_api_key
    ```

6. Place your trained model (`trained_model.h5`) in the `mlmodel` directory or update the `model_path` in the code to match the location of your model.

## Running the Application

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open your browser and go to:

    ```
    http://127.0.0.1:5000/
    ```

3. You can now upload ingredient images via the form, and the app will process the image, identify the ingredients, and show recipes with additional details like instructions, nutrition facts, TTS (Text-to-Speech), and YouTube videos.

## File Structure

