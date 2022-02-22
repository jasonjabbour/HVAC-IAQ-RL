#Import Gym Dependencies
import gym
from gym import Env
from gym.spaces import Discrete, Box

#Import Helpers
import numpy as np
import random
import time
import math

#number of steps for one episode
MAX_EPISODE_LEN = 1000 
ACH_LEVEL_KEY = [2,4]
C_AMBIENT = 400

class HVACEnv(Env):
    def __init__(self, GUI=False):
        # ADD ME: call plot if GUI is selected
        if GUI:
            pass

        self.GUI = GUI

        #History Queue Length
        self.history_steps_saved = 3 
        #CO2 History Queue
        self.CO2_history_queue = np.zeros([self.history_steps_saved])
        # #VOC History Queue
        # self.VOC_history_queue = np.zeros([self.history_steps_saved])
        # #PM2.5 History Queue
        # self.PM25_history_queue = np.zeros([self.history_steps_saved])
        #Temp History Queue
        # self.Temp_history_queue = np.zeros([self.history_steps_saved])
        #Occupancy History Queue
        self.Occupancy_history_queue = np.zeros([self.history_steps_saved])

        #Action Space: On or Off
        self.n_actions = 2 
        self.action_space = Discrete(self.n_actions)

        #Observational Space: CO2, VOC, PM2.5, Temp, and Occupancy, Time of Day
        #5*3=15 + 2 Time of Day (Hours, Minutes) = 17
        self.observation_space = Box(low=-1000, high=1000,shape=(2+2*self.history_steps_saved,))

        self.step_counter = 0

        #Call the reset function
        self.reset()


    def step(self, action):
        '''Step after a given action'''

        #Get the predicted ACH Level
        ACH_Level = ACH_LEVEL_KEY[action]

        #Ventilate for 15 mintues and return end CO2 Level
        #https://schools.forhealth.org/wp-content/uploads/sites/19/2020/08/Harvard-Healthy-Buildings-program-How-to-assess-classroom-ventilation-08-28-2020.pdf
        time_change = .25 #15 minutes
        # get CO2 Level to start (remember newest are appended to the end)
        CO2_start = self.CO2_history_queue[-1]
        CO2_end = (CO2_start - C_AMBIENT)*(math.e**(-(time_change)*ACH_Level)) + C_AMBIENT

        #Reward Function
        if CO2_end < 500:
            reward = 1
        else:
            reward = -1

        if (ACH_Level == 4) & (self.Occupancy_history_queue[-1] == 0):
            reward = -1

        done = False

        self.step_counter += 1
        if (self.step_counter > MAX_EPISODE_LEN):
            # Stop episode
            done = True

        #Step Time
        self.step_time()
        #Update Occupant Level
        occupancy = self.get_occupancy()
        #Update History Queues
        self.Occupancy_history_queue = self.update_queue(self.Occupancy_history_queue,occupancy)

        #Slowly start adding CO2 if occupant in room
        CO2_end += occupancy*random.randint(50,200)
        self.CO2_history_queue = self.update_queue(self.CO2_history_queue, CO2_end)        
        time_array = np.array([self.hour, self.minute])

        #Create the observation 
        self.observation = np.concatenate(((time_array, self.CO2_history_queue, self.Occupancy_history_queue)))

        print(f'Time: {self.hour} \tACH: {ACH_Level}  \tCO2: {int(CO2_end)}')

        info = {}
        return np.array(self.observation).astype(float), reward, done, info

    def render(self):
        '''Function required for OpenAI Gym env'''
        pass

    def reset(self):
        '''Reset environment each episode'''
        # print("--"*10,"New Episode","--"*10)

        #Reset episode counter
        self.step_counter = 0

        #Reset Start Time
        self.hour, self.minute = self.random_time()
        time_array = np.array([self.hour, self.minute])

        #Reset Queues
        #Get Occupancy
        self.Occupancy_history_queue = np.array([self.get_occupancy()]*self.history_steps_saved)
        self.CO2_history_queue = np.array([self.get_start_CO2()]*self.history_steps_saved)

        #Observations
        self.observation = np.concatenate((time_array, self.CO2_history_queue, self.Occupancy_history_queue))

        #return the initial observation
        return np.array(self.observation).astype(float)
    
    def step_time(self):
        '''Return an Hour (0-24) and Minutes in quarters (0-3 representing 00 15 30 45)'''
        self.hour = (self.hour+1)%24
        self.minute = (self.minute+1)%4
    
    def random_time(self):
        '''Generate a random hour and random minute'''
        return random.randint(0,23), random.randint(0,3)
    
    #FIX ME
    def get_occupancy(self):
        '''Use Noise and Motion to Predict Occupancy (0,1)'''
        #for now, I will randomly generate occupancy based on the time of day. Most likely between 8am to 5pm
        occupancy_choices = [0,1]
        prob_dist = [.9,.1]

        if self.hour >= 8 and self.hour<17:
            prob_dist = [.1,.9]
        
        return random.choices(occupancy_choices, prob_dist)[0]
    
    def get_start_CO2(self):
        '''Return a random start CO2 level in PPM'''
        return random.randint(700,1200)

    def update_queue(self, queue, value):
        '''A queue storing the history of occupancy'''
        #remove first observation 
        queue = queue[1:]

        #add new observation to end
        queue = np.append(queue,[value])
        return queue 
        
