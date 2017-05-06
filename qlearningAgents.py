# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import numpy as np
from game import *
from learningAgents import ReinforcementAgent
#from featureExtractors import *
import tensorflow as tf
from DQN import *
import random,util,math
from collections import deque

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self,environment, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)
    # set parameters

    params = {
      # Model backups
      'load_file': None,
      'save_file': None,
      'save_interval': 10000,

      # Training parameters
      'train_start': max([self.numTraining / 10, 32]),  # Episodes before training starts
      'batch_size': 32,  # Replay memory batch size
      'mem_size': 1000000,  # Replay memory size

      'discount': self.discount,  # Discount rate (gamma value)
      'lr': .0001,  # Learning reate
      'rms_decay': 0.9,  # RMS Prop decay
      'rms_eps': 1e-8,  # RMS Prop epsilon

      # Epsilon value (epsilon-greedy)
      'eps': 1,  # Epsilon start value
      'eps_final': .1,  # self.epsilon,  # Epsilon end value
      'eps_step': self.numTraining*10  # Epsilon steps between start and end (linear)
    }

    # params = {
    #   # Model backups
    #   'load_file': './saves/model-GridWorld_79596_682',
    #   'save_file': 'GridWorld',
    #   'save_interval': 10000,
    #
    #   # Training parameters
    #   'train_start': max([self.numTraining/10,32]),  # Episodes before training starts
    #   'batch_size': 32,  # Replay memory batch size
    #   'mem_size': 1000000,  # Replay memory size
    #
    #   'discount': self.discount,  # Discount rate (gamma value)
    #   'lr': .0000001,  # Learning reate
    #   'rms_decay': 0.9,  # RMS Prop decay
    #   'rms_eps': 1e-8,  # RMS Prop epsilon
    #
    #   # Epsilon value (epsilon-greedy)
    #   'eps': 1,  # Epsilon start value
    #   'eps_final': .1,#self.epsilon,  # Epsilon end value
    #   'eps_step': 1200000#self.numTraining*10  # Epsilon steps between start and end (linear)
    # }
    print("Initialise DQN Agent")

    # Load parameters
    self.params = params
    self.params['width'] = environment.gridWorld.grid.width
    self.params['height'] = environment.gridWorld.grid.height
    self.params['depth'] = 2
    self.params['num_training'] = self.numTraining
    self.params['numActions'] = len(environment.getPossibleActions(environment.getCurrentState()))
    # Start Tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    self.qnet = DQN(self.params)

    # time started
    self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
    # Q and cost
    self.Q_global = []
    self.cost_disp = 0

    # Stats
    self.cnt = self.qnet.sess.run(self.qnet.global_step)
    self.local_cnt = 0

    self.numeps = 0
    self.last_score = 0
    self.s = time.time()
    self.last_reward = 0.

    self.replay_mem = deque()
    self.last_scores = deque()
    self.env = environment
    self.registerInitialState()

  def registerInitialState(self):  # inspects the starting state
    state = None
    for x in range(self.params['width']):
      for y in range(self.params['height']):
        if self.env.gridWorld.grid.data[x][y] is 'S':
          state = (x,y)

    # Reset reward
    self.last_score = 0
    self.current_score = 0
    self.last_reward = 0.
    self.ep_rew = 0

    # Reset state
    self.last_state = None
    self.current_state = self.getStateMatrix(state)

    # Reset actions
    self.last_action = None

    # Reset vars
    self.terminal = None
    self.won = True
    self.Q_global = []
    self.delay = 0

    # Next
    self.frame = 0
    self.numeps += 1

  def getStateMatrix(self,state):
    # do walls
    walls = np.zeros((self.params['width'],self.params['height']))
    for x in range(self.params['width']):
      for y in range(self.params['height']):
        if self.env.gridWorld.grid.data[x][y] == '#':
          walls[x][y] = 1.
    # do character
    char = np.zeros((self.params['width'], self.params['height']))
    if state is 'TERMINAL_STATE':
      for x in range(self.params['width']):
        for y in range(self.params['height']):
          if self.env.gridWorld.grid.data[x][y] == self.last_reward:
            char[x][y] = 1.
    else:
      char[state[0]][state[1]] = 1.
    return np.stack((walls,char),axis=2)

  def actionToInt(self,action):
    # if action is 'exit':
    #   return random.choice([0,1,2,3])
    actionDict = {'north':0,'south':1,'east':2,'west':3}
    return actionDict[action]

  def actionToIntTrain(self,action):
    if action is 'exit':
      return random.choice([0,1,2,3])
    actionDict = {'north':0,'south':1,'east':2,'west':3}
    return actionDict[action]

  def intToAction(self,int):
    actionDict = {0:'north',1:'south',2:'east',3:'west'}
    return actionDict[int]

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    if action == 'exit':
      return self.env.gridWorld.grid.data[state[0]][state[1]]
    stateMat = self.getStateMatrix(state)
    # Exploit action
    Q_pred = self.qnet.sess.run(
      self.qnet.y,
      feed_dict={self.qnet.x: np.reshape(stateMat,
                                         (1, self.params['width'], self.params['height'], self.params['depth']))})[0]#,
                 #self.qnet.q_t: np.zeros(1),
                 #self.qnet.actions: np.zeros((1, 4)),
                 #self.qnet.terminals: np.zeros(1),
                 #self.qnet.rewards: np.zeros(1)})[0]
    return Q_pred[self.actionToInt(action)]

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    stateMat = self.getStateMatrix(state)
    # Exploit action
    self.Q_pred = self.qnet.sess.run(
      self.qnet.y,
      feed_dict={self.qnet.x: np.reshape(stateMat,
                                         (1, self.params['width'], self.params['height'], self.params['depth'])),
                 self.qnet.q_t: np.zeros(1),
                 self.qnet.actions: np.zeros((1, 4)),
                 self.qnet.terminals: np.zeros(1),
                 self.qnet.rewards: np.zeros(1)})[0]

    self.Q_global.append(max(self.Q_pred))
    a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))
    if len(a_winner) > 1:
      action = self.intToAction(
        a_winner[np.random.randint(0, len(a_winner))][0])
    else:
      action = self.intToAction(
        a_winner[0][0])

    return action

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """


    legalActions = self.getLegalActions(state)
    if legalActions == ('exit',):
      return 'exit'
    # e-greedy
    if util.flipCoin(self.params['eps']):
      # Save last_action
      action = random.choice(legalActions)
      self.last_action = self.actionToInt(action)
      return action
    else:
      # Save last_action
      action = self.getPolicy(state)
      self.last_action = action
      return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """


    #reward = self.getReward(nextState)

    # Process current experience reward
    self.current_score = reward
    self.last_score = reward
    self.last_reward = reward


    # Process current experience state
    self.last_state = np.copy(self.getStateMatrix(state))
    self.current_state = self.getStateMatrix(nextState)
    self.last_action = self.actionToIntTrain(action)
    self.terminal = False
    if nextState == 'TERMINAL_STATE':
      self.terminal = True

    self.ep_rew += self.last_reward

    # Store last experience into memory
    experience = (self.last_state, float(reward), self.last_action, self.current_state, self.terminal)
    self.replay_mem.append(experience)

    if len(self.replay_mem) > self.params['mem_size']:
      self.replay_mem.popleft()

    # Save model
    if (self.params['save_file']):
      if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
        self.qnet.save_ckpt('saves/model-' + self.params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
        print('Model saved')

    # Train
    self.train()

    # Next
    self.local_cnt += 1
    self.frame += 1
    self.params['eps'] = max(self.params['eps_final'],
                             1.00 - float(self.cnt) / float(self.params['eps_step']))

  def train(self):
    # Train
    if (self.local_cnt > self.params['train_start']):
      batch = random.sample(self.replay_mem, self.params['batch_size'])
      batch_s = []  # States (s)
      batch_r = []  # Rewards (r)
      batch_a = []  # Actions (a)
      batch_n = []  # Next states (s')
      batch_t = []  # Terminal state (t)

      for i in batch:
        batch_s.append(i[0])
        batch_r.append(i[1])
        batch_a.append(i[2])
        batch_n.append(i[3])
        batch_t.append(i[4])
      batch_s = np.array(batch_s)
      batch_r = np.array(batch_r)
      batch_a = self.get_onehot(np.array(batch_a))
      batch_n = np.array(batch_n)
      batch_t = np.array(batch_t)

      self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)

  def get_onehot(self, actions):
    """ Create list of vectors with 1 values at index of action in list """
    actions_onehot = np.zeros((self.params['batch_size'], 4))
    for i in range(len(actions)):
      actions_onehot[i][int(actions[i])] = 1
    return actions_onehot

  def getReward(self,state):
    try:
      reward = int(self.env.gridWorld.grid.data[state[0]][state[1]])
    except:
      reward = 0.0
    return reward

  def final(self):
    # Next
    #self.ep_rew += self.last_reward

    # Do observation
    self.terminal = True
    #self.observation_step(state)

    # Print stats
    log_file = open('./logs/' + str(self.general_record_time) + '-l-' + str(self.params['width']) + '-m-' + str(
      self.params['height']) + '-x-' + str(self.params['num_training']) + '.log', 'a')
    log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                   (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
    log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
    sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                     (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
    sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
    sys.stdout.flush()

