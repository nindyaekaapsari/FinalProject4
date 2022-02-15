import flask
from flask import request
import numpy as np
import pickle
import pandas as pd

scaler = pickle.load(open('model/scaler.pkl', 'rb'))
norm = pickle.load(open('model/normalize.pkl', 'rb'))
model = pickle.load(open('model/modelfp4.pkl', 'rb'))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return(flask.render_template('main.html'))
if __name__ == '__main__':
    app.run(debug=True)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    BALANCE = float(request.form['BALANCE'])
    BALANCE_FREQUENCY = float(request.form['BALANCE_FREQUENCY'])
    PURCHASES = float(request.form['PURCHASES'])
    INSTALLMENTS_PURCHASES = float(request.form['INSTALLMENTS_PURCHASES'])
    CASH_ADVANCE = float(request.form['CASH_ADVANCE'])
    PURCHASES_FREQUENCY = float(request.form['PURCHASES_FREQUENCY'])
    ONEOFF_PURCHASES_FREQUENCY = float(request.form['ONEOFF_PURCHASES_FREQUENCY'])
    CASH_ADVANCE_FREQUENCY = float(request.form['CASH_ADVANCE_FREQUENCY'])
    PURCHASES_TRX = float(request.form['PURCHASES_TRX'])
    CREDIT_LIMIT = float(request.form['CREDIT_LIMIT'])
    PAYMENTS = float(request.form['PAYMENTS'])
    MINIMUM_PAYMENTS = float(request.form['MINIMUM_PAYMENTS'])
    PRC_FULL_PAYMENT = float(request.form['PRC_FULL_PAYMENT'])
    TENURE = float(request.form['TENURE'])
    predict_list = [[BALANCE,BALANCE_FREQUENCY,PURCHASES,INSTALLMENTS_PURCHASES,CASH_ADVANCE,
            PURCHASES_FREQUENCY,ONEOFF_PURCHASES_FREQUENCY,CASH_ADVANCE_FREQUENCY,PURCHASES_TRX,
            CREDIT_LIMIT,PAYMENTS,MINIMUM_PAYMENTS,PRC_FULL_PAYMENT,TENURE]]
    predict = scaler.transform(predict_list)
    predict = norm(predict_list)
    feat_cols = ['BALANCE','BALANCE_FREQUENCY','PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE',
            'PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','CASH_ADVANCE_FREQUENCY','PURCHASES_TRX',
            'CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']
    predict = pd.DataFrame(predict,columns=feat_cols)
    prediction = model.predict(predict)
    output = {0: 'Klaster 0', 1: 'Klaster 1',2:'Klaster 2',3:'Klaster 3'}
    return flask.render_template('main.html', prediction_text='Customer termasuk dalam {}'.format(output[prediction[0]]))