import math
import random
import pylab as plt

def emulate(env, agent, maxSteps=math.inf):
  env.reset()
  agent.reset()
  replay = []
  done = False
  while (len(replay) < maxSteps) and not done:
    action = agent.process(env.state)
    state, reward, done, prevState = env.apply(action)
    replay.append((prevState, action, reward, state))
  return replay, done

def emulateBatch(testEnvs, agent, maxSteps=math.inf):
  replays = [[] for _ in testEnvs]
  for e in testEnvs: e.reset()
  
  steps = 0
  while (steps < maxSteps) and not all(e.done for e in testEnvs):
    steps += 1
    actions = agent.processBatch([e.state for e in testEnvs])
    for i, (e, action) in enumerate(zip(testEnvs, actions)):
      if not e.done:
        state, reward, _, prevState = e.apply(action)
        replays[i].append((prevState, action, reward, state))
        
  return [(replay, e.done) for replay, e in zip(replays, testEnvs)]

def plotData2file(data, filename):
  plt.clf()
  _, axes = plt.subplots(nrows=len(data))
  axes = axes if axes is list else [axes]
  for (chartname, series), axe in zip(data.items(), axes):
    for name, dataset in series.items():
      axe.plot(dataset, label=name)
    axe.title.set_text(chartname)
  plt.legend()
  plt.savefig(filename)
  plt.close()
  return
