import gym
import numpy as np
import time
import nasim

env = nasim.generate(num_hosts=30, num_services=5, num_os=3, num_processes=2, \
                    num_exploits=None, num_privescs=None, r_sensitive=10, r_user=10, \
                    exploit_cost=1, exploit_probs="mixed", privesc_cost=1, privesc_probs=1.0, \
                    service_scan_cost=1, os_scan_cost=1, subnet_scan_cost=1, process_scan_cost=1,\
                    uniform=False, alpha_H=2.0, alpha_V=2.0, lambda_V=1.0, restrictiveness=3, \
                    random_goal=True, base_host_value=1, host_discovery_value=1, \
                    seed=None, name=None, step_limit=1000000, address_space_bounds=None, yz_gen=True)
#env = gym.make("nasim:HugeGen-v0")
env = gym.wrappers.RecordEpisodeStatistics(env)

action_space_size = env.action_space.n

sps = []

for i in range(10):
    ob = env.reset()
    done = False
    num_sample = 0
    since = time.time()
    while not done:

        action = int(np.random.randint(low=0, high=action_space_size))
        ob, reward, done, info = env.step(action)
        num_sample += 1


        if done:
            print("===Done====")
            print(info)
            break
    time_elapsed = time.time()-since
    sps.append(num_sample/time_elapsed)

print("Aerage Sample per second:{}".format(np.mean(sps)))
print("Sample per second SD:{}".format(np.std(sps)))
print("Sample per second :{:.2f} +- {:.2f}".format(np.mean(sps), np.std(sps)))