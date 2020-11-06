from flask import Flask, request
from flask_cors import CORS,cross_origin
import joblib
import numpy as np
import os
app = Flask(__name__)
CORS(app)

@app.route('/iris',methods=['POST'])
@cross_origin() 
def predict_species():
    model = joblib.load('iris.model')
    req = request.values['param']
    input = np.array(req.split(','),dtype=np.float32).reshape(1,-1)
    predict_target = model.predict(input)
    if predict_target == 0 :
        return 'Setosa'
    elif predict_target == 1 :
        return 'Versiscolor'
    else:
        return 'Virginica'    

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port)