import gym
import keras
import numpy as np
import math
import random
import pickle

def get_model():
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.optimizers import Adam

    model = Sequential()
    model.add(Dense(24, input_dim=4))
    model.add(Activation('tanh'))
    model.add(Dense(48))
    model.add(Activation('tanh'))
    model.add(Dense(2))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer=Adam(lr=0.01, decay=0.01))

    return model

def choose_action(model, state, env, epsilon):
    # choose an action based on the observation
    sample = np.random.random_sample()
    if sample <= epsilon:
        return env.action_space.sample()
    return np.argmax(model.predict(state)[0])

env = gym.make('CartPole-v0')
model = get_model()

score = 0
epsilon = 1.0
epsilon_decay = 0.995
batch_size = 64
epsilon_min=0.05
max_mem = 256

records = []

for i_episode in range(2000):
    obs = env.reset()
    obs = obs.reshape((1, 4))
    done = False
    
    # run simulated game
    while not done:
        action = choose_action(model, obs, env, epsilon)

        new_obs, reward, done, info = env.step(action)
        new_obs = new_obs.reshape((1, 4))

        records.append((obs, new_obs, action, done))

        obs = new_obs
        if i_episode%100 == 0:
            env.render()
        score += 1

    # learn from the game
    x_train = []
    y_train = []
    minibatch = random.sample(records, min(len(records), batch_size))
    for obs, next_obs, action, done in minibatch:
        x_train.append(obs)
        # maximum reward for this action = 1 + maximum reward for next state
        y_target = model.predict(obs)
        maximum_future_reward = 1 + np.max(model.predict(next_obs)[0])
        # we only know about the action we took, so use the predicted for the other action
        y_target[0][action] = 1 if done else maximum_future_reward
        y_train.append(y_target)

    # try to make the agent forget the distant memory
    # if len(records) > max_mem:
    #     records = records[-max_mem:]

    x_train = np.array(x_train)
    x_train = x_train.reshape((x_train.shape[0], 4))
    y_train = np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], 2))

    model.fit(x_train, y_train, verbose=0)

    epsilon *= epsilon_decay
    epsilon = max(epsilon_min, epsilon)

    if i_episode%100 == 0:
        print('batch score: %s' % str(score/100))
        score = 0

model.save('model.h5')
pickle.dump(records, open('records.pkl', 'wb'))
    
    

