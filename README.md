
#### deep Q-learning implmented in pacman and the gridworld of the Berkeley CS188 Intro to AI codebase.

To start a training session from scratch run:  

~~~~
python3 gridworld.py -a q -k 1000
~~~~

-or-  

~~~~
python3 pacman.py -p PacmanDQN -n 6000 -x 5000 -l smallGrid
~~~~  

To run a pre-trained network, open qlearningAgents.py, in the \_\_init\_\_ section, comment out the active params dict and uncomment the inactive params dict, or replace it with something like the following:

~~~~
params = {
    # Model backups
    'load_file': './saves/model-GridWorld_79596_682',
    'save_file': 'GridWorld',
    'save_interval': 10000,

    # Training parameters
    'train_start': max([self.numTraining/10,32]),  # start training
    'batch_size': 32,  # Replay memory batch size
    'mem_size': 1000000,  # Replay memory size

    'discount': self.discount,  # Discount rate (gamma value)
    'lr': .0000001,  # Decreased Learning
    'rms_decay': 0.9,  # RMS Prop decay
    'rms_eps': 1e-8,  # RMS Prop epsilon

    # Epsilon value (epsilon-greedy)
    'eps': 1,  # Epsilon start value
    'eps_final': .1,#self.epsilon,  # Epsilon end value
    'eps_step': 1200000
    }
~~~~

Code was tested running on mac using python 3.6 and tensorflow 1.0.0, and windows using python 3.5 and tensorflow 1.0.1

Acknowledgements:  

Thanks to [tychovdo](https://github.com/tychovdo/PacmanDQN) and [devsisters](https://github.com/devsisters/DQN-tensorflow) for their excellent implementations of DQNs in tensorflow.


```python

```
