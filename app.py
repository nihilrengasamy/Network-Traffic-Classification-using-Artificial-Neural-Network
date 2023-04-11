from flask import Flask, render_template, request
import pickle
import numpy as np
import h5py
import keras
from sklearn.utils.validation import check_array


# model = pickle.load(open('iri.pkl', 'rb'))

model = keras.models.load_model("classifier.h5")

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['duration']
    data2 = request.form['total_fiat']
    data3 = request.form['total_biat']
    data4 = request.form['min_fiat']   
    data5 = request.form['min_biat']
    data6 = request.form['max_fiat']
    data7 = request.form['mean_biat']
    data8 = request.form['flowPktsPerSecond']
    data9 = request.form['flowBytesPerSecond']
    data10 = request.form['min_flowiat']
    data11 = request.form['mean_flowiat']
    data12 = request.form['std_flowiat']
    data13 = request.form['std_active']

    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13]],dtype = float)
    arr = arr.reshape((1,13))
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)














