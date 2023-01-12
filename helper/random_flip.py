import numpy as np
import matplotlib.pyplot as plt
host_num = 10
host_state = np.array(host_num * [1])
active_host_num = []
print(host_state)

def draw_random_normal_int(low:int, high:int):
    '''
    # https://stackoverflow.com/a/69738042
    '''
    # generate a random normal number (float)
    normal = np.random.normal(loc=0, scale=1, size=1)

    # clip to -3, 3 (where the bell with mean 0 and std 1 is very close to zero
    normal = -3 if normal < -3 else normal
    normal = 3 if normal > 3 else normal

    # scale range of 6 (-3..3) to range of low-high
    scaling_factor = (high-low) / 6
    normal_scaled = normal * scaling_factor

    # center around mean of range of low high
    normal_scaled += low + (high-low)/2

    # then round and return
    return int(np.round(normal_scaled))

for i in range(100000):
    """
    Parameters
    ----------
    shutdown_num: int
        How many host to be reset at each time, randomly drawn from `low` to `high`.
        
    
    """

    #idx = np.random.choice([1, 3], host_num, replace=False, p=[0.1, 0.9])
    #host_state[idx] = -host_state[idx]
    #print(host_state)
    shutdown_num = draw_random_normal_int(low=3, high=8)
    #print(np.random.choice(host_num, shutdown_num, replace=False))
    idx = np.random.choice(host_num, shutdown_num, replace=False)
    #out = np.argpartition(np.random.rand(N,12*4),M,axis=1)[:,:M]
    #print(idx.tolist())
    #print( host_state[idx])
    host_state[idx] = - host_state[idx]
    #print(np.count_nonzero(host_state == 1))
    active_host_num.append(np.count_nonzero(host_state == 1))
plt.plot(active_host_num)
plt.savefig("./figure")
print(np.mean(active_host_num))
print(np.std(active_host_num))


