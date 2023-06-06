import datetime
import jwt
from flask import Flask, request
import os
from flask import jsonify
from flasgger import Swagger
import pandas as pd
import pickle

rl = pickle.load(open('models/ExtraTreesRegressor_model_bon.pkl', 'rb'))

app = Flask(__name__)

template = {
    "swagger": "2.0",
    "info": {
        "title": "PMM - Production Prediction",
        "description" : "API for PMM Agriculure Production Prediction",
        "contact" : {
            "contact1": "Mor Ndour",
            "contact2": "Adja Fatou Gaye"
        },
        "Version": "0.0.1"
    }
}

@app.route('/')
def home():
    return "Hello there"

# app.config['SECRET_KEY'] = "bdeb"
# app.config['SWAGGER'] = {'title': 'API Loan', 'version': '1.0'}

# @app.route('/v1/acme/signup/')
# def get_token():
#     expiration_date = datetime.datetime.utcnow()+datetime.timedelta(seconds=600)
#     token = jwt.encode({'exp':expiration_date}, app.config['SECRET_KEY'], algorithm='HS256')
#     return token

# @app.route('/v1/acme/projet/<string:token>', methods=['GET'])
# def get_all_projet(token):
#     """Get a project by id
#                 ---
#                 parameters:
#                   - name: projet_id
#                     in: path
#                     type: integer
#                     required: true
#                     description: code du projet
#
#                 responses:
#                   200:
#                     description: Détail du projet
#                 """
#     try:
#         jwt.decode(token, key=app.config['SECRET_KEY'], algorithms=['HS256',])
#     except:
#         return jsonify({'error': "Token invalid"})
#
#     return jsonify({'emps' :  acme})

@app.route('/predict_file', methods=["POST"])
def batch_prediction():
    """This function takes a file as input which has sepal length, sepal width, petal length and petal width
    and returns an array of predictions based on the number of data points
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: Success
    """
    df_test = pd.read_excel(request.files.get("file"))
    print(df_test.head())
    prediction = rl.predict(df_test)
    df_test['PoidsNet'] = prediction
    #
    return str(df_test)
    # return  "Hello 2"





if __name__ =='__main__':

    port = int(os.environ.get('PORT', 5000))
    sawagger = Swagger(app, template= template)
    app.run(debug=True, host='0.0.0.0', port=port)