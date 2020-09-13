import pygame, sys
import numpy as np
from agent import Agent
import pandas as pd

class Pong():
        
    def __init__(self):
        self.size = (800,600)
        self.action = None
        self.BLACK = (0,0,0)
        self.WHITE = (255,255,255)
        self.RED = (255,0,0)
        self.GREEN = (0,255,0)
        self.BLUE = (0,0,255)
        
        self.done = False
        
        pygame.init()
        
        #Initializing the display window
        size = (800,600)
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("pong")
        
        #Starting coordinates of the paddle
        self.rect_x = 400
        self.rect_y = 580
        
        #initial speed of the paddle
        self.rect_change_x = 0
        self.rect_change_y = 0
        
        #initial position of the ball
        self.ball_x = 50
        self.ball_y = 50
        
        #speed of the ball
        self.ball_change_x = 5
        self.ball_change_y = 5
        
        self.score = 0
        self.reward = 0
        
        #draws the paddle. Also restricts its movement between the edges
        #of the window.
        #game's main loop  
                   
    def drawrect(self,screen,x,y):
        if x <= 0:
            x = 0
        if x >= 699:
            x = 699    
        pygame.draw.rect(screen,self.RED,[x,y,100,20])
                    
    def step(self, iterations):
        clock=pygame.time.Clock()
        return_array = np.empty((0, 6), float)
        preprocessed_array = np.array([[self.ball_x, self.ball_y, self.rect_x, self.rect_y]])
        batch_size = 24
        agent = Agent(batch_size)
        i = 0
        while i < iterations:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    i = iterations
            preprocessed_array = np.append(preprocessed_array, np.array([[self.ball_x, self.ball_y, self.rect_x, self.rect_y]]), axis=0)
            data = self.prepocessing(preprocessed_array)
            action = agent.act(data)

            #action = round(random.uniform(0, 1))

            if action == 0 and i % 100 == 0: #move right
                self.rect_change_x = 5
                self.rect_change_y = 0

                     
            if action == 1 and i % 100 == 0: #move left
                self.rect_change_x = -5
                self.rect_change_y = 0

                                       
            self.screen.fill(self.BLACK)
            self.rect_x += self.rect_change_x
            self.rect_y += self.rect_change_y
                        
            if self.rect_x > 785:   
                self.rect_x  = 785

            if self.rect_x < 0:   
                self.rect_x  = 0 
            
            self.ball_x += self.ball_change_x
            self.ball_y += self.ball_change_y 
                        
            #this handles the movement of the ball.
            if self.ball_x<0:
                self.ball_x=0
                self.ball_change_x = self.ball_change_x * -1
            elif self.ball_x>785:
                self.ball_x=785
                self.ball_change_x = self.ball_change_x * -1
            elif self.ball_y<0:
                self.ball_y=0
                self.ball_change_y = self.ball_change_y * -1
            elif self.ball_x>self.rect_x and self.ball_x<self.rect_x+100 and self.ball_y==565:
                self.ball_change_y = self.ball_change_y * -1
                self.score = self.score + 1
                self.reward = 1
                text = font.render("You hit it - nice!", True, self.WHITE)
                self.screen.blit(text,[600,100])
            elif self.ball_y>600:
                self.ball_change_y = self.ball_change_y * -1
                self.score = 0
                                     
            pygame.draw.rect(self.screen,self.WHITE,[self.ball_x,self.ball_y,15,15])
            
            #drawball(screen,self.ball_x,self.ball_y)
            self.drawrect(self.screen,self.rect_x,self.rect_y)
            
            #self.score board
            font= pygame.font.SysFont('Calibri', 15, False, False)
            text = font.render("score = " + str(self.score), True, self.WHITE)
            self.screen.blit(text,[600,100])
            
            #store values in an array 
            return_array = np.append(return_array, np.array([[action, self.ball_x, self.ball_y, self.rect_x, self.rect_y, self.reward]]), axis=0)
            self.reward = 0
            preprocessed_array = np.array([[self.ball_x, self.ball_y, self.rect_x, self.rect_y]])
                           
            pygame.display.flip()         
            clock.tick(10000) 
            i += 1
        text = font.render("Collected Data - wait while we train model...", True, self.WHITE)
        self.screen.blit(text,[600,100])
        return return_array
        pygame.quit()  
        sys.exit()

    def prepocessing(self, data):
        data = pd.DataFrame(data, columns=['ball_x', 'ball_y', 'rect_x', 'rect_y'])
        data['ball_direx'] = data['ball_x'].diff()
        data['ball_direy'] = data['ball_y'].diff()
        data['rect_direx'] = data['rect_x'].diff()
            #Remove incomplemt rows
        data = data.dropna(how = 'any')
        X_train = data[['ball_x', 'ball_y', 'rect_x', 'rect_y', 'ball_direx', 'ball_direy', 'rect_direx']].to_numpy()
        return X_train




