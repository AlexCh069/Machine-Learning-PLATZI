import joblib 
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)
@app.route('/predict', methods=['GET'])

def predict():
    X_test = np.array([7.344094877,7.223904962,1.494387269,1.478162169,0.830875158,0.612924099,0.385399252,0.384398729,2.097537994])
    prediction = model.predict(X_test.reshape(1,-1))
    return jsonify({'predictions': list(prediction)})

if __name__ == '__main__':
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)