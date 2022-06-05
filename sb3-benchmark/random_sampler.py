import gym
from matplotlib.pyplot import step
import numpy as np
import time
import nasim

# env = nasim.generate(num_hosts = 200, 
#                      num_os = 3,
#                      num_services = 10,
#                      num_exploits = 30,
#                      num_processes = 3,
#                      restrictiveness = 5,
#                      step_limit = 300000,
#                      yz_gen=True, save_fig=False)
env = gym.make("nasim:Small-v2")
env = gym.wrappers.RecordEpisodeStatistics(env)
print(env.get_score_upper_bound())
print(env.action_space.nvec)

sps = []
steps = []
for i in range(1):
    ob = env.reset()
    done = False
    num_sample = 0
    since = time.time()
    while not done:
        mask = env.get_action_mask()

        #action = int(np.random.randint(low=0, high=action_space_size))
        
        ob, reward, done, info = env.step(action)
        num_sample += 1

        if done:
            print("===Done====")
            episodic_steps = info['episode']['l']
            break
    time_elapsed = time.time()-since
    sps.append(num_sample/time_elapsed)
    steps.append(episodic_steps)

print("Aerage Sample per second:{}".format(np.mean(sps)))
print("Sample per second SD:{}".format(np.std(sps)))
print("Sample per second :{:.2f} +- {:.2f}".format(np.mean(sps), np.std(sps)))
print("Steps to solve: {:.1f} +- {:.1f}".format(np.mean(steps), np.std(steps)))