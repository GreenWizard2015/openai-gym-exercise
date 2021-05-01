from Pendulum.PendulumEnvironment import PendulumEnvironment
import time
from Utils import emulateBatch

def showAgentPlay(agent, speed=.01, env=PendulumEnvironment):
  env = env()
  env.reset()
  while not env.done:
    env.apply(agent.process(env.state))
    env.render()
    time.sleep(speed)
  
  env.hide()
  return

def testAgent(agent, memory, episodes, processor=None, env=PendulumEnvironment):
  testEnvs = [env() for _ in range(episodes)]
  for replay, isDone in emulateBatch(testEnvs, agent):
    replay = replay if processor is None else processor(replay)
    memory.addEpisode(replay, terminated=not isDone)

  return [x.score for x in testEnvs]

def trackScores(scores, metrics, levels=[.1, .5, .9]):
  if 'scores' not in metrics:
    metrics['scores'] = {}
    
  def series(name):
    if name not in metrics['scores']:
      metrics['scores'][name] = []
    return metrics['scores'][name]
  ########
  N = len(scores)
  orderedScores = list(sorted(scores, reverse=True))
  totalScores = sum(scores) / N
  print('Avg. test score: %.1f' % (totalScores))
  series('avg.').append(totalScores)
  
  for level in levels:
    series('top %.0f%%' % (level * 100)).append(orderedScores[int(N * level)])
  return