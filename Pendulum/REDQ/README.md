# Randomized Ensembled Double Q-Learning (REDQ)

Реализация идей из статьи ["Randomized Ensembled Double Q-Learning: Learning Fast Without a Model"](https://paperswithcode.com/paper/randomized-ensembled-double-q-learning-1).

Я не проводил дотошные эксперименты, но в другом проекте данный подход показал хорошие результаты. Главным достоинством REDQ является возможность намного большего количества итераций обучения без переоценки Q-значений, что позволяет намного быстрее восстанавливать оптимальные стратегии из replay buffer. 