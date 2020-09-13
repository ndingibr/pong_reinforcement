
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import numpy as np
import random
import pandas as pd
from joblib import load

from sklearn.preprocessing import StandardScaler

class Agent:
	def __init__(self, state_size, is_eval=False, model_name="pong.h5", scaler_name="scaler.bin"):
		self.state_size = state_size 
		self.action_size = 2 
		self.inventory = []
		self.model_name = model_name
		self.scaler_name = scaler_name
		self.is_eval = is_eval
		self.mini_batch = pd.DataFrame(columns=['action', 'ball_x', 'ball_y', 'rect_x', 'rect_y', 'reward',  'ball_direx', 'ball_direy', 'rect_direx'])

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = load_model("models/" + model_name) if is_eval else self._model()
		self.scaler = load("models/" + scaler_name)

	def _model(self):

		#The Feed-Forward Neural Network is used
		#in training the Keras model for the binary classification if the rectangle 
		#should move right or left
        
		#add layers to model
		model = Sequential()
		model.add(Dense(250, activation='relu', input_shape=(7,)))
		model.add(Dense(250, activation='relu'))
		model.add(Dense(250, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		#compile model using accuracy to measure model performance
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		return model

	def act(self, state):

        #This method is called by the came to 
        #determine the couse of action. It returns 0 for moving right and 1 for moving left
 
		X = self.scaler.transform(state)
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return round(random.uniform(0, 1))

		return self.model.predict_classes(X)[0][0]

	def expReplay(self, data):

        #The neural network is retrained at this point. Only positive events are considers
        #4 steps before the rectangle hit the ball. Those experiences are the one that are used to build the binary classification
        #so that the algorithm would remember them

		p = 4 #look back parameter
        
        #identify rows when the ball hit the rectangle and 4 events before that and savethem in a mini-batch
		for index, row in data.iterrows():
			if data.at[index, 'reward'] == 1:
				for i in range(p):
					v = index + i - p + 1  #index value to look into the past

					self.mini_batch = self.mini_batch.append({'action' : data.at[v , 'action'],
                                                    'ball_x' : data.at[v, 'ball_x'],
                                                    'ball_y' : data.at[v, 'ball_y'],
                                                    'rect_x' : data.at[v, 'rect_x'],
                                                    'rect_y' : data.at[v, 'rect_y'],
                                                    'reward' : data.at[v, 'reward'],
                                                    'ball_direx' : data.at[v, 'ball_direx'],
                                                    'ball_direy' : data.at[v, 'ball_direy'],
                                                    'rect_direx' : data.at[v, 'rect_direx']},
                                                   ignore_index=True)
            
            #get the features for trainin teh mean batch
			X_train = self.mini_batch[['ball_x', 'ball_y', 'rect_x', 'rect_y', 'ball_direx', 'ball_direy', 'rect_direx']].to_numpy()
			
            #get the target for training the mini batch
			y_train = self.mini_batch['action']
        
		print('training set size = ' + str(len(X_train)))
        
        #just making sure there is something in the mini batch
		if len(X_train) > 0:
            
            	#standardization
			self.scaler  = StandardScaler().fit(X_train)
			X_train = self.scaler.transform(X_train)

            #train the model on the mini-batch
			self.model.fit(X_train, y_train, epochs=300, verbose = 1, validation_split=0.2)
            
            #epsilon 0 - purely greedy. we start at 1 - which means we are completly exploring
            #the value gets less as we train, approaching 0 as we become more and more dependent on the trained model
			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay 
        

    
