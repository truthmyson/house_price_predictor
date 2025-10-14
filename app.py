from flask import Flask, render_template, request, jsonify
from database import init_db, save_prediction_data
import json
import joblib
import pandas as pd
import numpy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Initialize database
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert data for processing (all radio buttons are 1 for yes, 0 for no)
        features = {
            'area': float(data.get('area', 0)),
            'bedrooms': int(data.get('bedrooms', 0)),
            'bathrooms': float(data.get('bathrooms', 0)),
            'stories': int(data.get('stories', 0)),
            'mainroad': data.get('mainroad', 0),
            'guestroom': data.get('guestroom', 0),
            'basement': data.get('basement', 0),
            'hotwaterheating': data.get('hotwaterheating', 0),
            'airconditioning': data.get('airconditioning', 0),
            'parking': int(data.get('parking', 0)),
            'prefarea': data.get('prefarea', 0),
            'furnishingstatus': data.get('furnishingstatus', 'unfurnished')
        }
        
        # convert to dta frame
        features = pd.DataFrame(features, index=[0])

        # make prediction
        predicted_price = make_prediction(features)
        

        features['name'] = data.get('name', 'Anonymous')
        
        # Save to database
        save_prediction_data(features, predicted_price)
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'message': 'Prediction completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def make_prediction(features):
    """
    Feature Engineering and Model prediction function
    """

    # activate all preprocessors and model
    model = joblib.load('model_linear.pkl')

    # we will preprocess the data
    preprocessor = joblib.load('preprocessor.pkl')

    # create pipeline for incoming input
    _data = preprocessor.transform(features)

    # predict house price
    price = model.predict(_data)

    # we will load the pipeline for the target feature and reverse the scaling
    pipeline = joblib.load('target_preprocessor.pkl')

    price = pipeline.inverse_transform(price.reshape(-1,1))
    
    return round(float(price[0]), 2)



if __name__ == '__main__':
    app.run(port=2662, host='0.0.0.0')
