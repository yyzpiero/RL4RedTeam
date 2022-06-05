from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
# required defines a mandatory argument
# default defines a default value if not specified

parser = argparse.ArgumentParser()

parser.add_argument('-file_name',
                    type=str,
                    default="Medium-v02021-09-16-19-16-51",
                    help="defines file name")
parser.add_argument('-log_dir_path',
                    type=str,
                    default="content/runs",
                    help="defines log directory")
args = parser.parse_args()

val = 1

file_name = args.file_name
# file_name_list = [
#     "Medium-v02021-09-16-19-16-51", "MediumMultiSite-v02021-09-16-19-39-28"
# ]
log_dir_path = args.log_dir_path
'''
Manually Input Directory Paths
TODO: Support iterations as X-axis
'''

log_dir_path = "content/runs/DQN/MultiSeed/Tiny-v0/09-27-10-56"
log_dir_path_2 = "content/runs/DQN/MultiSeed/Tiny-v0/09-27-11-50"
run_var_list = ['seed-0', 'seed-1', 'seed-2', 'seed-3', 'seed-4']
all_data = pd.DataFrame([])
all_data_2 = pd.DataFrame([])

SMOOTH_PLOT = False
SMOOTH_RATIO = 2

for run_var in run_var_list:
    df = pd.DataFrame()
    df_2 = pd.DataFrame()
    event_base = EventAccumulator(log_dir_path + '/' + run_var)
    event_base_2 = EventAccumulator(log_dir_path_2 + '/' + run_var)
    event_base.Reload()
    _, step_nums, rewards = zip(*event_base.Scalars('test_total_reward_iter'))

    df['episodes'] = step_nums
    # if SMOOTH_PLOT == True:
    #     df['total_reward'] = smooth(np.array(rewards), SMOOTH_RATIO)
    # else:
    df['total_reward'] = np.array(rewards)
    df['model'] = run_var
    all_data = pd.concat([all_data, df], ignore_index=True)
    event_base_2.Reload()
    _, step_nums_2, rewards_2 = zip(
        *event_base_2.Scalars('test_total_reward_iter'))

    df_2['episodes'] = step_nums_2
    df_2['total_reward'] = np.array(rewards_2)
    df_2['model'] = run_var
    all_data_2 = pd.concat([all_data_2, df_2], ignore_index=True)

linestyle = ['-', '--', ':', '-.']
color = ['r', 'g', 'b', 'k']
label = ['algo1', 'algo2', 'algo3', 'algo4']
#all_data.melt(var_name='iteration', value_name='total_rewards')

#print(all_data)

sns.set(style="darkgrid", font_scale=1.5)
sns.lineplot(
    data=all_data,
    #hue='model',
    x="episodes",
    y="total_reward",
    ci=90)
sns.lineplot(
    data=all_data_2,
    #hue='model',
    x="episodes",
    y="total_reward",
    ci="sd")

plt.show()
