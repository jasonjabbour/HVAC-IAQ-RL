# Import required packages 
import datetime
from datetime import timedelta
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
import os
import time

# Connect to the API
url = "https://developer-apis.awair.is/v1/orgs/2970"
payload = {}
headers= {'x-api-key':'4iS73nI45Lkt9ydm8i9wb4BADCCKf1Y9'}
orgs = requests.request("GET", url, headers = headers, data = payload)

# Check API connection status
print(orgs.text.encode('utf8'))

# Define dictionaries for sensor ids and the desired variables (keyList) you want to extract

# sensor_ids = {"sensor_name": sensor_id}
# "sensor_name" is the sensor name as listed on the AWAIR dashboard (example: 211-omni_10, 211_wall_back)
# "sensor_id" is the set of 5 digits at the end of the sensor's Device UUID seen from the AWAIR Device Management page
sensor_ids = {"211-omni_10": 13138}

# Define the set of variables to extract
# keyList = ["pm25", "co2", "voc", "humid", "temp", "pm10_est", "score", "lux", "spl_a"]
keyList = ["pm25", "co2", "voc"]

# Define the desired time zone
# To look up other timezone codes: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
eastern = pytz.timezone('US/Eastern')

# Define the year of which you want to extract the data for
desired_year = 2022

# Define the month of which you want to extract the data for
desired_month = 1

# Define the start day for the range you want to extract the data from
start_day = 18

# Define the end day for the range you want to extract the data from
end_day = 19

for sensors, ids in sensor_ids.items():
    datadict = {}
    for i in keyList:
        datadict[i] = []

    for day in range(start_day,end_day):
        for hour in range(24):
                
            t = datetime.datetime(desired_year, desired_month, day, hour, 0, 0)
        
            x = t + timedelta(hours = 1)
            
            t = t.isoformat()
            x = x.isoformat()
            url = "https://developer-apis.awair.is/v1/orgs/2970/devices/awair-omni/{}/air-data/raw?from={}&to={}&limit=360&desc=false&fahrenheit=false".format(ids,t, x)
        
                
            devices = requests.request("GET", url, headers=headers, data = payload)
            a = devices.text.encode('utf8')
            
            a = json.loads(a)
            print(a)
            
            
            for i in range(len(a['data'])):
                for j in range(len(a["data"][i]["sensors"])):                        
                    if a["data"][i]["sensors"][j]['comp'] in datadict.keys():
                        print(a['data'][i]['timestamp'])
                        datadict[a["data"][i]["sensors"][j]['comp']].append([a['data'][i]['timestamp'], a["data"][i]["sensors"][j]['value']])
                                    
    
    for ikey, key in zip(range(len(datadict.keys())), datadict.keys()):
        column_names_key = ["timestamp", key]
        df_key = pd.DataFrame(datadict[key], columns = column_names_key)
        
        if ikey == 0:
            df_all_key = df_key.copy()
        
        elif ikey > 0:
            df_all_key = pd.merge(df_all_key, df_key, on='timestamp')
    
    
    # Insert and format dates
    final_file = df_all_key.reset_index()
    final_file = final_file.set_index(pd.DatetimeIndex(final_file['timestamp']))
    final_file = final_file.drop(['index'], axis=1)
    final_file = final_file.drop(['timestamp'], axis=1)
    # final_file = final_file[~final_file.index.duplicated()]
    
    # Change the selected timezone on first line of previous chunk if necessary
    # final_file.index = final_file.index.tz_convert(eastern)
    
    # Save to csv and output notification of completion
    final_file.to_csv('data_sensor_{}.csv'.format(sensors))
    print("file is saved")