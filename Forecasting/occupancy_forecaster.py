import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

TIME_INTERVAL = 15 #Minutes
START_DATE = '2021-09-01 00:00:00'
END_DATE = '2021-12-01 00:00:00'
PRODUCTION = True
NUM_TREES_IN_FOREST = 100
MODEL = None
LABEL_ENCODERS = {'Day':None,'Hour':None,'QuarterHour':None}

def get_motion_data():
    '''
    Load and Clean Motion Data
    
    motion_data.csv has two important columns:
    1. time: time value was recorded
    2. value: 0 or 1 if motion was detected at time t
    
    '''
    global LABEL_ENCODERS

    # Read Motion Data Assuming Running from HVAC-IAQ-RL Directory
    df = pd.read_csv('./Data/motion_data.csv', usecols=['time','value'])

    #Cast value column to a string 0 or 1
    df = df.astype({'value':int})

    #Remove unwanted part of time column
    df['time'] = df['time'].map(lambda x: str(x)[:19])
    #Cast time value into datetime (https://strftime.org/)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

    #Resample Data for every 15 Minutes, 
    time_interval_str = str(TIME_INTERVAL) + 'T'
    df = df.resample(time_interval_str, on='time', closed='left').sum()
    
    #Make sure the time column is not an index but actually called time
    df.reset_index(inplace=True)

    #Now convert sum of motion in 15 Minute interval to string 1 or 0 (motion detected yes/no)
    df = df.astype({'value':bool})
    df['value'] = df['value'].map({False:'0',True:'1'})

    #Create Column of Day of the Week
    df['Day'] = df['time'].dt.strftime('%A')
    #Create Hour Column
    df['Hour'] = df['time'].dt.strftime('%H')
    df = df.astype({'Hour':int})
    df = df.astype({'Hour':str})
    #Create Quarter Hour Column
    df['QuarterHour'] = df['time'].dt.strftime('%M')
    df = df.astype({'QuarterHour':int})
    df = df.astype({'QuarterHour':str})

    #Convert Strings to Numbers using Label Encoder
    day_lab_encoder = LabelEncoder()
    hour_lab_encoder = LabelEncoder()
    quarterhour_encoder = LabelEncoder()
    day_label = day_lab_encoder.fit_transform(df['Day'])
    hour_label = hour_lab_encoder.fit_transform(df['Hour'])
    quarterhour_label = quarterhour_encoder.fit_transform(df['QuarterHour'])

    #Save the encoders
    LABEL_ENCODERS['Day'] = day_lab_encoder
    LABEL_ENCODERS['Hour'] = hour_lab_encoder
    LABEL_ENCODERS['QuarterHour'] = quarterhour_encoder

    #Append Encoded Columns to dataframe
    df['DayEncoded'] = day_label
    df['HourEncoded'] = hour_label
    df['QuarterHourEncoded'] = quarterhour_label

    #Cut data and use only from start date time end date
    start_date_dt = datetime.strptime(START_DATE,'%Y-%m-%d %H:%M:%S')
    end_date_dt = datetime.strptime(END_DATE,'%Y-%m-%d %H:%M:%S') 
    df = df[(df['time'] >= start_date_dt) & (df['time'] <= end_date_dt)]

    if not PRODUCTION:
        print('Motion Dataframe Loaded.')
    
    return df

def build_random_forest_model(df):
    '''Built and Return Random Forest Model'''

    #Use the Encoded Columns as the Features
    X = df.iloc[:,5:].values
    #Use the Motion 0/1 as the Response Variable
    y = df.iloc[:,1].values

    #Split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=0)
    
    if not PRODUCTION:
        print(f'Train Shapes {X_train.shape, y_train.shape} and Test Shapes: {X_test.shape, y_test.shape}')

    #Built Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=NUM_TREES_IN_FOREST, random_state=0)
    rf_classifier.fit(X_train, y_train)

    if not PRODUCTION:
        #Predict
        y_pred = rf_classifier.predict(X_test)
    
        #Evaluate Model
        print(f'Confusion Matrix (TP, FP, FN, TN): \n\n{confusion_matrix(y_test, y_pred)}')
        # print(f'Classification Report: \n\n{classification_report(y_test, y_pred)}')
        print(f'Accuracy Score: \n\n{accuracy_score(y_test, y_pred)}')

    print('>> Random Forest Model has been trained. <<')

    return rf_classifier

def make_occupancy_prediction(Day, Hour, QuarterHour):
    '''Using Random Forest Model'''

    #Convert to encoded value
    Day = LABEL_ENCODERS['Day'].transform([Day])[0]
    Hour = LABEL_ENCODERS['Hour'].transform([str(int(Hour))])[0]
    QuarterHour = LABEL_ENCODERS['QuarterHour'].transform([str(int(QuarterHour))])[0]

    occupancy_prediction = MODEL.predict([[Day, Hour, QuarterHour]])

    return occupancy_prediction[0]

def run_occupancy_forecaster(production=PRODUCTION):
    '''Run this function if calling occupancu forecaster from outside script then start calling make_occupancy_prediction()'''
    global MODEL, PRODUCTION
    
    PRODUCTION = production

    # Load the motion data
    motion_df = get_motion_data()

    #Build Random Forest Model
    rf_model = build_random_forest_model(motion_df)

    #save the model
    MODEL = rf_model

def plot_week():
    '''Plots the prediction of a week'''

    days_lst = ['Monday', 'Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    days_lst_abr = ['M', 'T','W','Th','F','Sat','Sun']
    quarterhour_lst = ['00','15','30','45']
    occupancy_output_list_of_dicts = []

    #Predict the occupancy for a week
    for day in range(len(days_lst)):
        for hour in range(24):
            for quarterhour in range(len(quarterhour_lst)):
                #Create new empty dictionary and initialize variables
                occupancy_output_dict = {}
                name = ''
                end_hour = ''
                color = ''

                #Make prediction
                occupancy_output = make_occupancy_prediction(days_lst[day],hour,int(quarterhour_lst[quarterhour]))

                #Name the event
                if int(occupancy_output):
                    name = 'occupied'
                    color = '29EE65' #Green
                else:
                    name = 'vacant'
                    color = '#D0D0D0' #Gray

                #Create time interval (remember to wrap around) 
                if quarterhour_lst[quarterhour] == '45':
                    end_hour = str((hour+1)%24)
                else:
                    end_hour = str(hour)
                
                #Since event must start before it ends
                if not ((str(hour) == '23') and (quarterhour_lst[quarterhour] == '45')):

                    #Save required dictionary for pdfschedule
                    occupancy_output_dict['name'] = name
                    occupancy_output_dict['days'] = days_lst_abr[day]
                    
                    #Time must be in 19:30-20:30 format
                    occupancy_output_dict['time'] = str(hour)+':' + quarterhour_lst[quarterhour] +'-'+ end_hour+':'+quarterhour_lst[(quarterhour+1)%4]
                    occupancy_output_dict['color'] = color

                    #Add dictionary to list of dictionaries
                    occupancy_output_list_of_dicts.append(occupancy_output_dict)
    
    #Save output predictions to yaml file
    with open('plot_week.yaml','w') as file:
        documents = yaml.dump(occupancy_output_list_of_dicts, file)

    #Pass yaml file to pdfschedule to generate plot
    #Run the following command: pdfschedule plot_week.yaml
   
if __name__ == "__main__":
    run_occupancy_forecaster(False)

    #Make a prediction (Day, Hour, QuarterHour)
    print('\n---- Example Predictions ----')
    print('Example Prediction (Wednesday at 4:15pm):', make_occupancy_prediction('Wednesday',16,15))
    print('Example Prediction (Sunday at 9:00pm):', make_occupancy_prediction('Sunday',21,0)) 

    #Plot the prediction of a week
    plot_week()


# --- Helpful Tools ---
# print(df.dtypes)
# print(df.time.dtype)
# pd.set_option('display.max_rows',None)
# print(df.head(120))
# print(lab_encoder.classes_)
#Assumption: not online learning. Cannot update the model with new data. Must retrain the model

