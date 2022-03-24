import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import manifold,datasets

def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))


def run_inits_steps(envs, rollout, render, n_init_steps):
    ############################################
    # env.reset()

    for _ in range(n_init_steps):
        #env.action_space.seed(seed=seed)
        #action = torch.from_numpy(env.action_space.sample())
        action = torch.from_numpy(np.random.randint(0, envs.action_space.n, size=(envs.num_envs,)))
        next_state, _, done, _ = envs.step(action)
        next_state = next_state.data.cpu().numpy().tolist()
        rollout.save_observation(next_state)

        if render:
            envs.render()

        # if done:
        #     env.reset()

    rollout.update_obs_normalization_param(rollout.obs_buffer.observations)
    rollout.obs_buffer.clear_memory()

    return rollout

def test_episode(env, agent, action_dim=2, render=False):
    ############################################
    state = env.reset()
    done = False
    total_reward = 0
    eps_time = 0
    if agent.coverage:
        coverage_hist = np.ones(action_dim)
    #state_vec = []
    #action_vec = []
    while not done:
        
        if agent.mask_type == "Hard":
            action_re_mask = action_mask
            action_re_mask_prob = np.array(action_re_mask )/sum(action_re_mask)
        else:
            action_re_mask = np.ones(action_dim)
            action_re_mask_prob = np.array(action_re_mask )/sum(action_re_mask)
        if agent.coverage:
            coverage_hist_norm = coverage_hist/np.sum(coverage_hist)
        #action = int(agent.act(state, action_re_mask_prob, coverage_hist=coverage_hist_norm))
        action = int(agent.act(state))
        #state_vec.append(state)

        next_state, reward, done, info = env.step(action)
        if agent.coverage:
            coverage_hist[action] = coverage_hist[action] + 1

        #action_vec.append(action)      
        
        if agent.mask_type is not None:
            action_mask = info['action_mask']
            agent.action_mask = torch.tensor(np.array(action_mask, dtype=bool), requires_grad=False).to(device=agent.device)
        
        #agent.action_mask = torch.tensor(action_mask)

        eps_time += 1
        total_reward += reward

        # if training_mode:
        #     agent.save_eps(state.tolist(), float(action), float(
        #         reward), float(done), next_state.tolist(), coverage_hist_norm.tolist())
        #     agent.save_observation(next_state)

        state = next_state

        if render:
            env.render()

        # if training_mode:
        #     if t_updates % agent.rnd_step_update == 0:
        #         agent.update_rnd()
        #         t_updates = 0
        #     #torch.cuda.empty_cache()

        if done:
            #print(total_reward)

            return total_reward, eps_time#, t_updates#, agent.entropy_loss, agent.vf_loss, agent.coverage_loss#, info['num_disrupted'], info['num_privesc'], coverage_hist_norm




def run_episode_coverage(env, agent, state_dim, action_dim, render, training_mode, t_updates ,n_step_update):
    ############################################
    state = env.reset()
    done = False
    total_reward = 0
    eps_time = 0
    # state_vec = []
    # action_vec = []
    coverage_hist = np.ones(action_dim)
    if agent.mask_type is not None:
        action_mask = env.get_action_mask()
        agent.action_mask = torch.tensor(np.array(action_mask, dtype=bool), requires_grad=False).to(device=agent.device)
    ############################################

    while not done:
        
        if agent.mask_type == "Hard":
            action_re_mask = action_mask
            action_re_mask_prob = np.array(action_re_mask )/sum(action_re_mask)
        else:
            action_re_mask = np.ones(action_dim)
            action_re_mask_prob = np.array(action_re_mask )/sum(action_re_mask)
        
        coverage_hist_norm = coverage_hist#/np.sum(coverage_hist)
        action = int(agent.act(state, action_re_mask_prob, coverage_hist=coverage_hist_norm))
        
        # state_vec.append(state)
        # action_vec.append(action)
        next_state, reward, done, info = env.step(action)
        coverage_hist[action] = coverage_hist[action] + 1      
        
        if agent.mask_type is not None:
            action_mask = info['action_mask']
            agent.action_mask = torch.tensor(np.array(action_mask, dtype=bool), requires_grad=False).to(device=agent.device)
        
        #agent.action_mask = torch.tensor(action_mask)

        eps_time += 1
        t_updates += 1
        total_reward += reward

        if training_mode:
            agent.save_eps(state.tolist(), float(action), float(
                reward), float(done), next_state.tolist(), coverage_hist_norm.tolist())
            agent.save_observation(next_state)

        state = next_state

        if render:
            env.render()

        if training_mode:
            if t_updates % n_step_update == 0:
                agent.update_rnd()
                #agent.update_ppo()
                t_updates = 0
            #torch.cuda.empty_cache()
        if done:
           
            break
         
    print(total_reward)
    return total_reward, eps_time, t_updates, agent.entropy_loss, agent.vf_loss, agent.coverage_loss, coverage_hist
#, info['num_disrupted'], info['num_privesc'], coverage_hist_norm



def rollout_steps(env, agent, rollout_runner, n_steps, render, training_mode, state, global_rollout_step):
    ###########################################
    """we have to re-write this part. To run n_steps of """
    
    ############################################
    #state = env.reset()
    
    # total_reward = 0
    # eps_time = 0
    # state_vec = []
    # action_vec = []
    ############################################

    for iter in range(n_steps):
        global_rollout_step += env.num_envs
        """
        action (num_envs)
        state  (num_envs x shape_obs)
        """   
        action = agent.act(state)
        # state_vec.append(state)
        # action_vec.append(action)
        next_state, reward, done, infos = env.step(action)
        #next_state_buffer = next_state.data.cpu().numpy().tolist()
    
        for idx, info in enumerate(infos):
            if 'episode' in info.keys():
                #print(f"global_step={ global_rollout_step}, episode_reward={info['episode']['r']}")
                #writer.add_scalar("charts/episodic_return", info['episode']['r'], global_step)
                #writer.add_scalar("charts/episode_curiosity_reward", curiosity_rewards[step][idx], global_step)
                break
        # eps_time += 1
        # t_updates += 1
        # total_reward += reward

        if training_mode:
            rollout_runner.save_eps(state.tolist(), action.data.cpu().numpy(), reward.view(-1).data.cpu().numpy(), done, next_state.tolist())
            rollout_runner.save_observation(next_state.data.cpu().numpy().tolist())

        state = next_state

        if render:
            env.render()

        # if training_mode:
        #     if t_updates % n_step_update == 0:
        #         agent.update_rnd()
        #         #agent.update_ppo()
        #         t_updates = 0
            #torch.cuda.empty_cache()
       
         
        
    return rollout_runner, next_state, done, global_rollout_step