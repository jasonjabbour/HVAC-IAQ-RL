# HVAC-IAQ-RL

## A Reinforcement Learning Approach Towards Developing Smart HVAC Systems
Utilizing environmental data including CO2 Levels, Occupancy, and Time of Day to predict ACH Levels

### Usage
-  `HVAC_env.py`: Custom OpenAI Gym environment
-  `HVAC_train.py`: Driver to create a gym environment, train model using the PPO algorithm, and save the model within the `Training` directory
-  `HVAC_enjoy.py`: Load a trained model and simulate a smart HVAC system. 

### Train Model
 ``` commandline
 python3 HVAC_train.py
 ```
 
 ### Load and View Model
 ``` commandline
 python3 HVAC_enjoy.py
 ```


