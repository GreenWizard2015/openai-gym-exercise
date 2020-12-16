# Utils.Networks

Набор вспомогательных/специализированных обёрток и функций для нейронных сетей.

## **GhostNetwork.py**

В Reinforcement Learning часто применяют приём, когда делается копия сети и используется для расчёта target-a обучаемой сети, а затем новые веса переносятся в копию. Класс GhostNetwork облегчает данную задачу, скрывая в себе процесс работы с "призрачной" сетью.

В конструктор класса или в метод **update** можно передать функцию "смешивания" весов. Если передать `'hard'`, то в "призрачную" сеть будут полностью перенесены веса из обучаемой сети.

**WeightsLinearMix(tau)** - простая линейная интерполяция весов. **tau** определяет долю переноса весов изм новой сети в старую.

Пример использования:

```
ghostNetwork = GhostNetwork(model, mixer=WeightsLinearMix(0.1))
....
ghostNetwork.update('hard') # полностью обновляем копию
for epoch in range(TRAIN_EPISODES):
    if 0 == (epoch % 5):
      ghostNetwork.update() # обновляем копию ЧАСТИЧНО

    states, actions, rewards, nextStates, nextStateScoreMultiplier = memory.sampleBatch(
      batch_size=BATCH_SIZE, maxSamplesFromEpisode=16
    )
    actions = ACTIONS.toIndex(actions)
    # используем копию для предсказаний
    futureScores = ghostNetwork.predict(nextStates).max(axis=-1) * nextStateScoreMultiplier
    targets = ghostNetwork.predict(states)
    targets[np.arange(len(targets)), actions] = rewards + futureScores * GAMMA

    lazyNN.ghostNetwork(states, targets, epochs=1, verbose=0) # обучаем основную сеть
```

## **LazyNetwork.py**

Основано на идее описанной в [этом видео](https://www.youtube.com/watch?v=rvr143crpuU), которое основано на [Accelerating Deep Learning by Focusing on the Biggest Losers](https://arxiv.org/abs/1910.00762).

Мотивация данной идеи:

- нейронные сети быстрее выполняют операцию предсказания, а обучение намного более дорогая операция.

- обучение на хорошо усвоенных примерах не вносит новых "знаний" и мешает обучению более сложным примерам.

Пример:

```
lazyNNProvider = LazyNetwork(
  model,
  batchSize=BATCH_SIZE,
  patience=3 * BATCH_SIZE, # выбирать из 3 батчей
  fitArgs={'epochs': 1, 'verbose': 0}, # доп. параметры обучения
  loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE) # loss для КАЖДОГО сэмпла
)
............
with lazyNNProvider as lazyNN:
  for _ in range(TRAIN_EPISODES):
    states, actions, rewards, nextStates, nextStateScoreMultiplier = memory.sampleBatch(
      batch_size=BATCH_SIZE, maxSamplesFromEpisode=16
    )
    actions = ACTIONS.toIndex(actions)
    
    futureScores = lazyNN.predict(nextStates).max(axis=-1) * nextStateScoreMultiplier
    targets = lazyNN.predict(states)
    targets[np.arange(len(targets)), actions] = rewards + futureScores * GAMMA

    lazyNN.fit(states, targets)
```

`fit` выполняется при достижении лимита `patience` или при выходе из блока `with` (если достаточно сэмплов).

Тестировал на DQN с/без bootstrap - увы, результаты плохие. Подобный подход я применял в задаче сегментации и там он сильно улучшал результаты, поэтому возможно я не правильно реализовал/применил подход.