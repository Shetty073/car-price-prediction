from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
lr_model = joblib.load('./model/car_price_lr.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.form
        prediction = lr_model.predict(np.array([[payload['year'], payload['present_price'], payload['kms_driven'], payload['owner']]]))
        context = {
            'prediction': f'{prediction[0]:.2f}',
        }
        return render_template('predict.html', context=context)
    except Exception as e:
        context = {
            'error': f'{e}'
        }
        return render_template('predict.html', context=context)
        
