
import gym
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

env = gym.make('CartPole-v0')

obs = env.reset()
obs = obs.reshape((1, 4))

done = False
score = 0
# run simulated game
while not done:

    action = np.argmax(model.predict(obs)[0])
    obs, _, done, _ = env.step(action)
    obs = obs.reshape((1, 4))

    env.render()

    score += 1

print('survived %d steps' % score)
