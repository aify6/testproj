from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


# import and load pickle files
model = pickle.load(open('models/forest_fire_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        # get the data from the form
        data = request.form.to_dict()
        # convert to dataframe
        data = pd.DataFrame(data, index=[0])
        # convert to float
        data = data.astype(float)
        # scale the data
        data_scaled = scaler.transform(data)
        # make prediction
        prediction = model.predict(data_scaled)
        # return the prediction
        if prediction[0] == 1:
            result = 'Fire Detected'
        else:
            result = 'No Fire Detected'
        return render_template('home.html', prediction=result, names=data.columns.tolist())
    else:
        return render_template('home.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)