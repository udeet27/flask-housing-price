from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoder
with open('model/model.pkl', 'rb') as f:
    saved = pickle.load(f)
    weights = saved['weights']
    bias = saved['bias']
    encoder = saved['encoder']

def predict(features):
    return np.dot(features, weights) + bias

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        size = int(request.form['size'])
        pincode = request.form['pincode']

        # Encode pincode
        location_encoded = encoder.transform([[pincode]])
        features = np.hstack([[bedrooms, bathrooms, size], location_encoded[0]])
        prediction = round(predict(features), 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
