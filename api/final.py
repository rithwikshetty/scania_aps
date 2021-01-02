import joblib
import pandas as pd
from flask import Flask, jsonify, request
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from lightgbm import LGBMClassifier
import flask
import time

import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello Rithwik"

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict' , methods=['POST'])
def predict():
    path = request.form.to_dict()
    print(path)   

    aps_data = pd.read_csv(path["inFile"] , na_values=["na"])

    start = time.time()
	# Load imputation models
    median_imputer = joblib.load('..\\models\\median_imputer.pkl')
    mice_imputer = joblib.load('..\\models\\mice_imputer.pkl')

    # Drop features with max null values
    aps_data = aps_data.drop(['br_000','bq_000','bp_000','bo_000','ab_000','cr_000','bn_000','cd_000'] , axis=1)
 
    # Specify features whose missing values are imputed using Median Imputer
    median_features = ['ak_000','ca_000','dm_000','df_000','dg_000','dh_000','dl_000','dj_000','dk_000','eb_000','di_000','ac_000','bx_000','cc_000']
    
    # Median Imputation
    aps_data[median_features] = median_imputer.transform(aps_data[median_features])
    
    # MICE Imputation
    aps_data = pd.DataFrame(data = mice_imputer.transform(aps_data) , columns= aps_data.columns )
    
    # Load GBDT Model
    model = joblib.load("..\\models\\gbdt_model.pkl")
    
    # Predict class label 
    aps_data['class'] = model.predict(aps_data)
    end = time.time()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    aps_data.to_csv( path['outFile'] + '\\output_' + str(timestr) + '.csv' , index=False )
    
    return 'Process Complete. Please check Output Directory. Total Process Time:'+str(end-start)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

