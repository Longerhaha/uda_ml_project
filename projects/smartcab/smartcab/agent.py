import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=True, epsilon=1.0, alpha=0.3):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        #self.Q = dict({('green','left','RightCar_allow_move','LeftCar_allow_move','OncomingCar_allow_move'):dict({None:0.0,'right':0.0,'left':0.0,'forward':0.0})})# Create a initial Q-table which will be a dictionary of tuples
        self.Q = dict({('ergent_degree_1','green','left',True,True,True):dict({None:0.0,'right':0.0,'left':0.0,'forward':0.0})})
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        
        #self.state_old = tuple(['green','left','RightCar_allow_move','LeftCar_allow_move','OncomingCar_allow_move'])
        self.state_old = tuple(['ergent_degree_1','green','left',True,True,True])
        self.action_old = 'left'
        self.reward_old = 4.0
        self.decrese_rate_1 = 0.97
        #self.decrese_rate_2 = 0.94
    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        #self.epsilon = self.epsilon-0.05
        #self.epsilon = self.epsilon*math.exp(-1*self.decrese_rate_1)
        
        '''
        if self.epsilon >= 0.6:
            self.epsilon = self.epsilon*math.exp(-1*self.decrese_rate_1)
            self.alpha = 0.1
        elif self.epsilon >= 0.03:
            self.epsilon = self.epsilon*math.exp(-1*self.decrese_rate_2)
            self.alpha = 0.5
        else:
            self.epsilon = self.epsilon-0.001
            self.alpha = 0.9'''
        if testing:
            self.epsilon = 0
            self.alpha = 0 
        else:
            self.epsilon = self.epsilon*self.decrese_rate_1
            
        return None
       
    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        #print 'waypoint is %s'%waypoint
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline
        #print 'deadline is %d'%deadline
        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent 
        
        #state = (inputs['light'],waypoint,inputs['right'],inputs['left'],inputs['oncoming'])
        '''
        if deadline >= 14:
            reach_des_ergent_degree = 'ergent_degree_1'
        elif deadline >= 6:
            reach_des_ergent_degree = 'ergent_degree_2'
        else:
            reach_des_ergent_degree = 'ergent_degree_3'
        '''  
        if deadline >= 12:
            reach_des_ergent_degree = 'ergent_degree_1'
        else:
            reach_des_ergent_degree = 'ergent_degree_2'
          
        if waypoint == 'forward':
            allow_left = True
            allow_right = True
            if inputs['right'] == 'forward' or inputs['left'] == 'forward':
                allow_forward = False
            else:
                allow_forward = True
        elif waypoint == 'left':
            allow_forward = True
            allow_right = True
            if inputs['right'] == 'forward' or inputs['left'] == 'forward' or inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right':
                allow_left = False
            else:
                allow_left = True
        else:
            allow_forward = True
            allow_left = True
            if inputs['left'] == 'forward':
                allow_right = False
            else:
                allow_right = True
        
        state = (reach_des_ergent_degree,inputs['light'],waypoint,allow_forward,allow_left,allow_right)
        
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        
        maxQ = max(self.Q[state].itervalues())
        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if self.learning:
            if not(state in self.Q):
                self.Q.setdefault(state,dict({None:0.0,'right':0.0,'left':0.0,'forward':0.0}))
            else:
                pass 
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        if not self.learning:
            action = random.choice([None, 'forward', 'left', 'right'])
        else:
            #self.random_value = random.randint(1,100)
            #print 'epsilon_value ',int(self.epsilon * 100),'random_value ',self.random_value
            if random.random() <= self.epsilon:
                action = random.choice([None, 'forward', 'left', 'right'])
                #print 'random_action'
            else:
                bestaction = []
                max_val = self.get_maxQ(state)
                for key in self.Q[state].keys():
                    if self.Q[state][key] == max_val:
                        bestaction.append(key) 
                
                action = random.choice(bestaction)
                #print 'max_Q_action'
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning:
            self.Q[self.state_old][self.action_old] = round( (1-self.alpha) * self.Q[self.state_old][self.action_old] + self.alpha * (self.reward_old + self.get_maxQ(state)) ,3)
            
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward 
        self.learn(state, action, reward)   # Q-learn
        self.state_old = state   # keep state of last step
        self.action_old = action  # keep action of last step
        self.reward_old = reward  # keep reward of last step
        #print self.Q
        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent,enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env,update_delay=0.001,log_metrics=True,optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=50,tolerance=0.005)
    # cd C:\Users\Administrator\Desktop\git-learn\machine-learning\projects\smartcab
    # python ./smartcab/agent.py
if __name__ == '__main__':
    run()
