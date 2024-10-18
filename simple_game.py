
import numpy as np

class SimpleGame:
    def __init__(self):
        self.state = np.zeros((80, 80, 3), dtype=np.uint8)
        self.done = False

    def reset(self):
        self.state = np.zeros((80, 80, 3), dtype=np.uint8)
        self.done = False
        return self.state

    def step(self, action):
        reward = 0
        if action == 0:  # Example action
            reward = 1
        elif action == 1:  # Example action
            reward = -1
        elif action == 2:  # Example action
            reward = 0
        self.done = np.random.rand() > 0.95  # Randomly end the game
        return self.state, reward, self.done, {}
