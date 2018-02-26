
#%%
import numpy as np
from hmmlearn import hmm




states = np.array(["box 1", "box 2", "box3"])
n_states = len(states)

observations = np.array(["red", "white", "green"])
n_observations = len(observations)

start_probability = np.array([0.2, 0.4, 0.4])

transition_probability = np.array([
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]
])

emission_probability = np.array([
  [0.5, 0.2,0.3],
  [0.4, 0.1,0.5],
  [0.7, 0.1,0.2]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability

#%%
seen = np.array([[0,2,1]]).T
logprob, box = model.decode(seen, algorithm="viterbi")
print(logprob)
print(box)
print("The ball picked:", ", ".join(map(lambda x: observations[x], seen.reshape(-1))))
print("The hidden box:", ", ".join(map(lambda x: states[x], box)))

#%%
box2 = model.predict(seen)
print("The ball picked:", ", ".join(map(lambda x: observations[x], seen.reshape(-1))))
print("The hidden box", ", ".join(map(lambda x: states[x], box2)))


#%%
model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
X2 = np.array([[0,2,0,1],[0,0,0,1],[1,0,2,1]])
model2.fit(X2)
print (model2.startprob_)
print (model2.transmat_      )
print (model2.emissionprob_)
print (model2.score(X2))
model2.fit(X2)
print (model2.startprob_)
print (model2.transmat_)
print (model2.emissionprob_)
print (model2.score(X2))
model2.fit(X2)
print (model2.startprob_)
print (model2.transmat_)
print (model2.emissionprob_)
print (model2.score(X2))