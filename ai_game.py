
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from simple_game import SimpleGame

class AIGame:
    def __init__(self):
        self.model = self.create_model()
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(3, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(3)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (80, 80))
        return np.reshape(resized, (1, 80, 80, 1))

    def train(self, game):
        state = self.preprocess_frame(game.reset())
        state = np.stack([state] * 4, axis=1)
        done = False
        while not done:
            action = self.act(state)
            next_state, reward, done, _ = game.step(action)
            next_state = self.preprocess_frame(next_state)
            next_state = np.append(state[:, 1:], np.expand_dims(next_state, 0), axis=1)
            self.remember(state, action, reward, next_state, done)
            state = next_state
            if len(self.memory) > 32:
                self.replay(32)

if __name__ == "__main__":
    game = SimpleGame()
    ai_game = AIGame()
    ai_game.train(game)
