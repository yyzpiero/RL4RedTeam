import gym
import sys

# sys.path.append("/home/yangyizhou/Codes/rl-replications/NASim-test/")
import nasim


def EnvironmentBuild(name: str) -> gym.envs:
    #print(name[0:5])
    if name[0:5] == "nasim":
        print("----------Using NASIM ENV-------")
        #return gym.make(name)
        return gym.make(name)
    else:
        print("----------Using GYM ENV-------")
        return gym.make(name)  #.unwrapped