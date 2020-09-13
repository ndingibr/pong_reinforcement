#import libraries

#import our game
from env import Pong
import pandas as pd
#import agent
from agent import Agent

#for saving the training and scaler parameters
from joblib import dump

#modue used to write the game
import pygame
    
env = Pong()

#we will train the game for 500 episodes
episode_count = 500
batch_size = 24
agent = Agent(batch_size)

print('Watch out for the game auto playing in the popup window. the game stops here and there to retrain the algorithm')

def data_engineering(data):

    #adds variables to find the direction of movemnt of the ball and the rectangle
    data['ball_direx'] = data['ball_x'].diff()
    data['ball_direy'] = data['ball_y'].diff()
    data['rect_direx'] = data['rect_x'].diff()
    
    #Remove incomplemt rows
    data = data.dropna(how = 'any')
    return data

for e in range(episode_count):
    
    #play the game
    numpy_data = env.step(10000) 
    
    #get the coordinates and the rewards
    data = pd.DataFrame(numpy_data, columns=['action', 'ball_x', 'ball_y', 'rect_x', 'rect_y', 'reward'])
    
    #engineer more features
    data = data_engineering(data)
    
    #replay the episode and learn from it
    agent.expReplay(data)
    
    #only save the model when the episode # is divisible by 100
    if e % 100 == 0:
        agent.model.save("models/" + agent.model_name)
        dump(agent.scaler, "models/" + agent.scaler_name, compress=True)
 
pygame.quit() 


    


