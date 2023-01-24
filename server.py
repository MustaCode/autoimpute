# Import flask and datetime module for showing date and time
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file, Response, send_from_directory, make_response
import datetime
import csv
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imputation.imputex import imputex
from imputation.autoimputex import autoimputex
from pymongo import MongoClient, errors
from mongopass import mongopass
import certifi
from bson.json_util import dumps, loads
import json
from bson.objectid import ObjectId

# import numpy as np

x = datetime.datetime.now()

# Initializing flask app
app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})

# MongoDB
try:
    print('Connecting to Database...')
    mongo_client = MongoClient(mongopass, tlsCAFile=certifi.where())
    # db = mongo_client.imputex
    # collection = db.datasets
    # db = mongo_client['imputex']
    # collection = db['datasets']
    print("Connection to Database Successful!")
except errors.InvalidURI:
    print('Error Connecting to Database')

# print(mongo_client.imputex.datasets.insert_one({'name': 'Musta'}).inserted_id)



# client = MongoClient(mongopass)
# db = client.imputex
# datasetsColl = db.datasets


# Route for seeing a data
# @app.route('/data')
# def get_time():

#     # Returning an api for showing in reactjs
#     return {
#         'Name': "geek",
#         "Age": "22",
#         "Date": x,
#         "programming": "python"
#     }

# Route for seeing a data


@app.route('/impute', methods=['POST', 'GET'])
def impute():

    # with mongo_client.imputex.datasets.watch() as stream:
    #     for change in stream:
    #         print(change)

    # content_type = request.headers.get('Content-Type')
    dataset = request.files['dataset']
    df = pd.read_csv(dataset, dtype='unicode')
    df_bench = pd.read_csv('benchmark/data_original_lite.csv', dtype='unicode')

    # df.to_csv('exports/data.csv', encoding='utf-8')

    # print(df.info())

    # print(df.isna().sum().sum(), " Total missing values before imputation 1")


    df_imputed = imputex(df, df_bench)
    df_imputed.to_csv('exports/data.csv', encoding='utf-8', index=False)

    # return imputed_dataset

    
    # return {"status": "OK"}

    # csvList = '\n'.join(','.join(row) for row in csvList)
    # si = StringIO()
    # cw = csv.writer(si)
    # cw.writerows(df)
    # output = make_response(si.getvalue())
    # output.headers["Content-Disposition"] = "attachment; filename=export.csv"
    # output.headers["Content-type"] = "text/csv"
    # return output

    rows = len(df)

    try:
        exact_path = r"exports/data.csv"
        response = send_file(exact_path, as_attachment=True)
        # response.set_header('x-myapp-rows', rows)
        return response
        # return send_file(exact_path, as_attachment=True)
        # return '{} {}'.format(send_file(exact_path, as_attachment=True), rows)
    except Exception as e:
        return str(e)

    # return send_file(path, download_name='df', as_attachment=True)
    # try:
    #    path = f'./csv_files/{df}'
    # except Exception as e:
    #     return str(e)

    # return df.to_html(header="true", table_id="table")

    # return render_template('table.html', tables=[df.to_html()], titles=[''])
    # return render_template('table.html', tables=[df.to_html()], titles=df.columns.values)
    # return render_template('simple.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
    # print(df.to_string())
    # return "Dataset is: {0}".format(dataset)

# @app.route('/export', methods=['POST', 'GET'])
# def export():
#     try:
#         exact_path = r"exports/data.csv"
#         return send_file(exact_path, as_attachment=True)
#     except Exception as e:
#         return str(e)

# # Running app
# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/autoimpute', methods=['POST', 'GET'])
def autoimpute():
    # db = mongo_client.imputex
    # # col = db.datasets

    # cursor = db.datasets.find()

    # list_cur = list(cursor)
    # json_data = dumps(list_cur)

    # df = pd.read_json(json_data, dtype='unicode')

    # # df = df.avg.replace({np.nan: None}, inplace=True)

    # df = df.replace('', np.nan)  
    # df = df.replace('nan', np.nan)  

    # df_imputed = autoimputex(df)

    # print(df_imputed)

    # for i in results:
    #     print(i)


    # START LISTENING TO THE COLLECTION INSERTS
    with mongo_client.imputex.datasets.watch([{
        '$match': {
            'operationType': { '$in': ['insert'] }
        }
    }]) as stream:
        for change in stream:

            # STOP THE WATCH IF GENDER EQUALS STOP
            if change['fullDocument']['Diagnosis'] == 'stop':
                break

            # print(change)
            
            # GET ALL RECORDS
            db = mongo_client.imputex
            collection = db.datasets
            cursor = collection.find()
            # CONVERT THE CURSOR TO LIST
            list_cur = list(cursor)
            # CONVERT THE LIST TO JSON
            json_data = dumps(list_cur)
            
            # CONVERT JSON TO DATAFRAME
            df = pd.read_json(json_data, dtype='unicode')

            # CONVERT EMPTY STRINGS AND NAN TO NUMPY MISSING VALUES
            df = df.replace('', np.nan)  
            df = df.replace('nan', np.nan)  

            # FILTER ALL RECORDS WITH MISSING VALUES
            df_miss = df[df.isnull().any(axis=1)]

            # APPLY IMPUTEX TO ALL RECORDS
            df_imputed = autoimputex(df)

            # ITERATE THROUGH ID RECORDS WITH MISSING VALUES ONLY
            for id in df_miss['_id']:
                # convert dictionary string to dictionary
                # id_json = eval(id)

                # REPLACE ' WITHN " TO AVOID ERRORS
                id_str = id.replace("\'", "\"")

                # CONVERT STRING ID TO DICTIONARY
                id_json = json.loads(id_str)

                # GET THE ID NUMBER ONLY FROM THE DICTIONARY
                id_filter = id_json['$oid']

                # GET IMPUTED RECORDS THAT EQUALS THE ID OF MISSING RECORDS
                sel_row = df_imputed[(df_imputed == id).any(axis=1)]

                # DELETE THE ID COLUMN
                del sel_row['_id']

                # CONVERT JSON TO DICTIONARY WITH INDEX
                sel_row_dict = sel_row.to_dict(orient='index')

                # print(sel_row_dict[sel_row.first_valid_index()])
                
                # MATCH THE ID FOR UPDATE QUERY
                filter = { '_id': ObjectId(id_filter) }

                # VALUES TO BE UPDATED
                newvalues = { "$set": sel_row_dict[sel_row.first_valid_index()] }

                # UPDATE ONE MV RECORD WITH IMPUTED RECORD 
                collection.update_one(filter, newvalues)

            # print(df_miss)

            # df_imputed = autoimputex(df)

            # print(df_imputed)

            # if change['fullDocument']['name'] == 'musta':
            #     break
            # return "New Record!!!"
    return "Auto Imputation is on!"


@app.route('/data', methods=['GET'])
def data():
    db = mongo_client.imputex

    cursor = db.datasets.find({}, {'_id': False })

    list_cur = list(cursor)
    json_data = dumps(list_cur)

    return json_data


@app.route('/insertdata', methods=['POST'])
def insertdata():
    db = mongo_client.imputex

    Diagnosis = request.form['Diagnosis']
    Age = request.form['Age']
    PTGENDER = request.form['PTGENDER']
    PTEDUCAT = request.form['PTEDUCAT']
    PTETHCAT = request.form['PTETHCAT']
    PTRACCAT = request.form['PTRACCAT']
    PTMARRY = request.form['PTMARRY']
    CDRSB = request.form['CDRSB']
    ADAS11 = request.form['ADAS11']
    ADAS13 = request.form['ADAS13']
    autoImputeOn = request.form['autoImputeOn']

    # print(age)

    db.datasets.insert_one({
        'Diagnosis': Diagnosis,
        'Age': Age,
        'PTGENDER': PTGENDER,
        'PTEDUCAT': PTEDUCAT,
        'PTETHCAT': PTETHCAT,
        'PTRACCAT': PTRACCAT,
        'PTMARRY': PTMARRY,
        'CDRSB': CDRSB,
        'ADAS11': ADAS11,
        'ADAS13': ADAS13,
    })

    print('autoImputeOn', autoImputeOn)

    # IF NO MV, RETURN THE DATA WITHOUT UPDATE
    if Diagnosis != '' and Age != '' and PTGENDER != '' and PTEDUCAT != '' and PTETHCAT != '' and PTRACCAT != '' and PTMARRY != '' and CDRSB != '' and ADAS11 != '' and ADAS13 != '':
        db = mongo_client.imputex
        collection = db.datasets
        cursor = collection.find({}, {'_id': False })
        list_cur = list(cursor)
        json_data = dumps(list_cur)
        return json_data


    # LISTEN TO UPDATES
    if autoImputeOn == 'true':
        with mongo_client.imputex.datasets.watch([{
            '$match': {
                'operationType': { '$in': ['update'] }
            }
        }]) as stream:
            for change in stream:
                print('true')
                db = mongo_client.imputex
                collection = db.datasets
                cursor = collection.find({}, {'_id': False })
                list_cur = list(cursor)
                json_data = dumps(list_cur)
                # print('autoImputeOn', autoImputeOn)
                return json_data
    else:
        print('false')
        db = mongo_client.imputex
        collection = db.datasets
        cursor = collection.find({}, {'_id': False })
        list_cur = list(cursor)
        json_data = dumps(list_cur)
        return json_data

    # list_cur = list(cursor)
    # json_data = dumps(list_cur)

    # return 'success'

@app.route('/stopautoimpute', methods=['POST'])
def stopautoimpute():
    db = mongo_client.imputex

    # data = request.form
    Diagnosis = 'stop'

    # print(age)

    db.datasets.insert_one({
        'Diagnosis': Diagnosis,
    })

    db.datasets.delete_many({'Diagnosis': 'stop'})

    
    # list_cur = list(cursor)
    # json_data = dumps(list_cur)

    return 'success'