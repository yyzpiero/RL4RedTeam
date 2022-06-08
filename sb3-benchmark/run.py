import Algo

model = Algo.on_policy.ppo

model.learn()

model.save