from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Area=int(request.form['Area'])
        Park=int(request.form['Parking facility'])
        BUILDTYPE=int(request.form['BUILDTYPE'])
        Utility=int(request.form['Utility'])
        Street=int(request.form['Street'])
        Salecondition=int(request.form['Sale condition'])
        Mzzone=int(request.form['MZzone'])
        Intsqft=int(request.form['INTsqft'])
        Year = int(request.form['YearBUILD'])
        Houselife=int(request.form['Houselife'])
        X_test=[Area,Park,BUILDTYPE,Utility,Street,Salecondition,Mzzone,Intsqft,Year,Houselife]
        X_test = np.array(X_test).reshape((1,-1))
        prediction=model.predict(X_test)
        output=np.exp(prediction)
        if output<0:
            return render_template('index.html',prediction_texts="Please enter correct values")
        else:
            return render_template('index.html',prediction_text="You Can buy The house at ${}".format(output)+"/")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

