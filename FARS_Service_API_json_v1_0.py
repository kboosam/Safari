# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:10:06 2017

@author: kboosam
"""

'''
Current version notes: Works on flask. Tested with postman on local and it is working well. 

No Authorization implemented
No logging implementd
Exceptions are printed out, not logged
Requests are not tracked with unique ID. commented out.

IMP: @@@@turn off debug before deploying in production

'''


# Importing libraries

import pandas as pd
from flask import Flask, jsonify, request
#from flask_restful import Api
from flask_cors import CORS
import logging
import numpy as np

'''

Function to load the model pickle based on input filename 

'''
def load_model(filename):
    
    import pickle as dill
    loadedmodel = dill.load(open(filename, 'rb'))
    
    return loadedmodel
### END OF FUNCTION load_model
##############################################################

'''
Function to build the x (model inputs) with values extracted from json

'''
    
def create_x(req):
    
        '''    
        dataset structure for the model variables along with index
        1	>>>	SEX
        2	>>>	AGE
        3	>>>	PER_TYP
        6	>>>	BODY_TYP
        7	>>>	Veh_Age
        12	>>>	Snow
        15	>>>	ROUTE
        16	>>>	Weather
        17	>>>	veh_speed
        18	>>>	state
        19	>>>	spdlimit
        20	>>>	road_feat
                       
        '''
                           
        '''
        sample JSON strcuture
        {
        
        'GENDER': ,    #{'F' : 0, 'M' : 1}
        'AGE': ,
        'PASSTYPE': 1,  # awlays set this to 1 for now..
        'VEHTYPE': ,    #{'2D Sedan':0, '4DSedan':1, 'C pickup':2, 'C SUV':3, 'L Pickup':4, 'MC':5, 'Minivan':6 }
        'VEHAGE': ,
        'ROUTE': ,      #road type. should be from Google Roads API; 
                        {'Interstate':1, 'U.S. Highway':2, 'State Highway':3, 'County Road':4, 'Local Street':6}
        'WEATHER': ,    # Open weatherAPI. or radio buttons; {'Clear':0, 'Cloudy':1, 'Fog':2, 'Rain':3, 'Snow/Freeze':4, 'Strong Wind':5}
        'VEHSPEED': ,
        'STATE': ,      #geoacoding. should be from Google Roads API; 
        'ROADFEAT': ,   #road condition. May be from Google Roads API, need to check; Otherwise, we will need a widget. 
                        {'Non-junction': 1, 'Intersection':2, 'Entrance/Exit Ramp':5, 'Railway Grade Crossing':6, 'Through Roadway':18}
        'SPEEDLIM':     # from google roads API
         
        }
        '''
        
        ### populate snow impact (annual fall in inches) by state
        statecd = [1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,
                 26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,53,54,55,56]
        
        snow_data = [41,1892,8,132,0,485,1029,513,0,18,0,488,625,658,886,373,318,0,
                     1570,513,1113,1298,1372,23,432,968,658,554,1544,419,244,3145,193,1300,
                     699,198,76,716,859,13,1115,160,38,1427,2062,262,127,1575,1293,2322,1800]
        #snowdf = pd.DataFrame(index=range(51), data=snow_data, columns=['snow'])
        
        snowdf = pd.DataFrame(index=list(range(51)), data=statecd, columns=['stcd'])
        snowdf.loc[:,'snow'] = pd.Series(snow_data, index=snowdf.index)
        
                
        '''
        Populating the X values from the json request object
        
        '''
                
        x = []
        x.append(req['GENDER'])
        x.append(req['AGE'])
        x.append(req['PASSTYPE'])
        x.append(req['VEHTYPE'])
        x.append(req['VEHAGE'])
        #x.append(snowdf.iloc[req['STATE'],0]) 
        x.append(snowdf.loc[snowdf['stcd']==req['STATE'],'snow'].iloc[0])
        #print('snow:', snowdf.loc[snowdf['stcd']==req['STATE'],'snow'].iloc[0])
        x.append(req['ROUTE'])
        x.append(req['WEATHER'])
        x.append(req['VEHSPEED'])
        x.append(req['STATE'])
        x.append(req['ROADFEAT'])
        spd_lim = req['SPEEDLIM']
        
        
        '''
        An exam[ple x values and encoding maps for front end to send appropriate values      
        
        x = [1, #gender {'F' : 0, 'M' : 1}
             60, # Age
             1, #driver or passenger  {'Passenger':0, 'Driver':1}
             2, #vehicle type {'2D Sedan':0, '4DSedan':1, 'C pickup':2, 'C SUV':3, 'L Pickup':4, 'MC':5, 'Minivan':6 }
             10, #veh age in yrs {'Interstate':1, 'US Highway':2, 'St Highway':3, 'County Rd':4, 'Local St':5}
             0, #snow
             1, #route {'Interstate':1, 'US Highway':2, 'St Highway':3, 'County Rd':4, 'Local St':5}
             0, #weather {'Clear':0, 'Cloudy':1, 'Fog':2, 'Rain':3, 'Snow/Freeze':4, 'Strong Wind':5}
             65, #veh speed
             34, # state code
             0 # road features
             ]
        
        spd_lim = 65
        '''
        x.append(x[1]*x[8]) # Response control factor
        x.append(x[8]/spd_lim)  # speeding factor
        return x
#### END OF FUNCTION create_x
###################################################################

app = Flask(__name__)
CORS(app) ## cross origin resource whitelisting..

'''if app.config['CORS_ENABLED'] is True:
    CORS(app, origins="http://127.0.0.1:5100", allow_headers=[
        "Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
        supports_credentials=True)
'''


@app.route('/predict/safari_api', methods=['POST','GET'])
def safari():
    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    logging.basicConfig(level=logging.DEBUG)
    print('\n\n Started processing request...\n\n')
    
    try:
        #req_json = str(request.get_json())
        #req = pd.read_json("test_json.txt", typ='series')
        req = request.json
        
        #print('REQUID: ', req['REQUID'], '\n\n')
        print('Request: ', req, '\n\n')
        
    except Exception as e:
        print(e)
        #return "empty request!!!" ## temp for debug

    if any(req.values()): #any values in the request?  
        x = create_x(req)
    else:
        print('Empty Request!!!','\n\n')
        return(jsonify({'STATUS':400, 
                        'REASON': 'Empty request!!'}))
    
    ### Random Forest classification model        
    modelfile = 'FARS-RFC Model.sav'
        
    try:
        print('Loading the model....')
        model = load_model(modelfile)
        print('Predicting the probability....')
        x = np.array(x)
        x = x.reshape(1,-1) ## version 0.19 onwards scikit-learn expects 2D array. Original model was built with 0.18, hence this work around
        #print(x, type(x))
        
        y_prob = model.predict_proba(x)
        print('completed..','\n\n')
        stat_code = 100
        reason = 'successful'
        prob = round(y_prob[0,1],2)
    except Exception as e:
        stat_code = 500
        reason = 'Oops, something went wrong, looks like it got tired!!'
        prob = 99
        print(e) #printing all exceptions to the log
        
    finally:    
        resp = {'INCPROB': prob,
                'STATUS': stat_code,
                'REASON': reason}
    
        print("#### Response :", resp, '\n\n')
    
    return jsonify(resp)


if __name__ == '__main__':
   print('started running safari')
   app.run(debug=True,port=5100) #turnoff debug for production deployment
