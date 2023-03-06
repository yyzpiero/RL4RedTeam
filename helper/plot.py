#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-19


import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.font_manager import FontProperties
import os
""" For windows machines, load fonts 
#myfont=FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')
#sns.set(font=myfont.get_name())
"""

class Painter:
    def __init__(self, load_csv, load_dir=None):
        if not load_csv:
            self.data = pd.DataFrame(columns=['episodic_return','globl', 'Method'])
        else:
            self.load_dir = load_dir
            if os.path.exists(self.load_dir):
                print("==正在读取{}。".format(self.load_dir))
                self.data = pd.read_csv(self.load_dir).iloc[:,1:] # csv文件第一列是index，不用取。
                print("==读取完毕。")
            else:
                print("==不存在{}下的文件，Painter已经自动创建该csv。".format(self.load_dir))
                self.data = pd.DataFrame(columns=['episodic_return', 'global_step', 'Method'])
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.hue_order = None

    def setXlabel(self,label): self.xlabel = label

    def setYlabel(self, label): self.ylabel = label

    def setTitle(self, label): self.title = label

    def setHueOrder(self,order):
        """设置成['name1','name2'...]形式"""
        self.hue_order = order

    def addData(self, dataSeries, method, smooth = True):
        if smooth:
            dataSeries = self.smooth(dataSeries)
        size = len(dataSeries)
        for i in range(size):
            dataToAppend = {'episodic_return':dataSeries[i],'global_step':i+1,'Method':method}
            self.data = self.data.append(dataToAppend,ignore_index = True)

    def drawFigure(self):
        sns.set_theme(style="darkgrid")
        sns.set_style(rc={"linewidth": 1})
        print("==正在绘图...")
        #data=pd.melt(df, ['x'])
        sns.relplot(data = self.data, kind = "line", x = "global_step", y = "episodic_return", hue= "Method", hue_order=None)
        #sns.relplot(data = pd.melt(self.data, ["global_step"]), kind = "line", x = "global_step", y = "episodic_return")
        # sns.lmplot(data = self.data, scatter=False, x = "global_step", y = "episodic_return",
        #              ci=95)
        plt.title(self.title,fontsize = 12)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        print("==绘图完毕！")
        plt.show()
        plt.savefig("test.png")

    def saveData(self, save_dir):
        self.data.to_csv(save_dir)
        print("==已将数据保存到路径{}下!".format(save_dir))

    def addCsv(self, add_load_dir):
        """将另一个csv文件合并到load_dir的csv文件里。"""
        add_csv = pd.read_csv(add_load_dir).iloc[:,1:]
        self.data = pd.concat([self.data, add_csv],axis=0,ignore_index=True)

    def deleteData(self,delete_data_name):
        """删除某个method的数据，删除之后需要手动保存，不会自动保存。"""
        self.data = self.data[~self.data['Method'].isin([delete_data_name])]
        print("==已删除{}下对应数据!".format(delete_data_name))

    def smoothData(self, smooth_method_name,N):
        """对某个方法下的reward进行MA滤波，N为MA滤波阶数。"""
        begin_index = -1
        mode = -1  # mode为-1表示还没搜索到初始索引， mode为1表示正在搜索末尾索引。
        for i in range(len(self.data)):
            if self.data.iloc[i]['Method'] == smooth_method_name and mode == -1:
                begin_index = i
                mode = 1
                continue
            if mode == 1 and self.data.iloc[i]['global_step'] == 1:
                self.data.iloc[begin_index:i,0] = self.smooth(
                    self.data.iloc[begin_index:i,0],N = N
                )
                begin_index = -1
                mode = -1
                if self.data.iloc[i]['Method'] == smooth_method_name:
                    begin_index = i
                    mode = 1
            if mode == 1 and i == len(self.data) - 1:
                self.data.iloc[begin_index:,0]= self.smooth(
                    self.data.iloc[begin_index:,0], N=N
                )
        print("==smoothing{} in {}th order!".format(smooth_method_name,N))

    @staticmethod
    
    def smooth(data,N=5):
        """
        TODO: This function should also make sure that `x` are the same,
               otherwiswe
        
        """

        n = (N - 1) // 2
        res = np.zeros(len(data))
        for i in range(len(data)):
            if i <= n - 1:
                res[i] = sum(data[0:2 * i+1]) / (2 * i + 1)
            elif i < len(data) - n:
                res[i] = sum(data[i - n:i + n +1]) / (2 * n + 1)
            else:
                temp = len(data) - i
                res[i] = sum(data[-temp * 2 + 1:]) / (2 * temp - 1)
        return res
    
    def round_to_nearest(x):
        return round(x, -int(math.floor(math.log(x, 1000))))

def record_single_csv(logdir_path="runs/CartPole/CartPole-v1-1",
                      smooth=(True, 5),
                      interp_step=(True, 5000, 1000), 
                      step_str="global_step", 
                      value_str="episodic_return", 
                      method_str="Clap"):
    event_accumulator = EventAccumulator(logdir_path)
    event_accumulator.Reload()
    scalar_str = "charts/" + value_str
    events = event_accumulator.Scalars(scalar_str)
    
    x = [x.step for x in events]
    y = [x.value for x in events]
    
    if interp_step[0]==True:

        ceil_length_x = int(interp_step[1])*math.ceil(int(x[-1])/int(interp_step[1]))
        #print(ceil_length_x)
        #print(round(ceil_length_x/1000))
        x_interp = np.arange(1, ceil_length_x, round(ceil_length_x/int(interp_step[2])))       
        y1_interp = np.interp(x_interp, x, y)

        if smooth[0] == True:
            y1_interp = np.convolve(y1_interp, np.ones(smooth[1])/smooth[1], mode='same')
        

        df = pd.DataFrame({step_str: x_interp[:-5], value_str: y1_interp[:-5]})
    else:
        if smooth[0] == True:
            y = np.convolve(y, np.ones(smooth[1])/smooth[1], mode='same')
        df = pd.DataFrame({step_str: x[:-1], value_str: y[:-1]})

    df = df.assign(Method=method_str)
    return df

def get_string_before_last_slash(string):
    if string[-1]=="/":
        string = string[:-1]
    last_slash_index = string.rfind('/')
    if last_slash_index == -1:
        return string
    else:
        return string[last_slash_index+1:]
    

def tb_log_to_csv(logdir_path="runs/CartPole/CartPole-v1-1", 
                      single_log=True, 
                      smooth=(True, 5),
                      interp_step=(True, 5000, 1000), 
                      step_str="global_step", 
                      value_str="episodic_return", 
                      method_str="Clap",
                      save_path=(True, "split", "./csv/CartPole")):
        
   

    if single_log==True:
        df = record_single_csv(logdir_path=logdir_path,
                               smooth=smooth,
                               interp_step=interp_step,
                               step_str=step_str,
                               value_str=value_str,
                               method_str=method_str)
        if save_path[0] == True:
            df.to_csv(save_path + get_string_before_last_slash(logdir_path))
        return df
    else:
        dir_path = logdir_path
        df_t = None
        # Get all subdirectories in the current working directory
        subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        
        for logdir in subdirs:
            
            df = record_single_csv(os.path.join(logdir_path, logdir))
            
            if save_path[0] == True and save_path[1] == "split":
                df.to_csv(os.path.join(save_path[-1], logdir))
            
            if df_t is not None:
                df_t = pd.concat([df_t, df], axis=0, ignore_index=True)
            else:
                df_t = df

        if save_path[0] == True and save_path[1] == "aggr":
            save_dir = os.path.join(save_path[-1], get_string_before_last_slash(logdir_path))
            if not os.path.exists(save_path[-1]):
                os.makedirs(save_path[-1])
            df_t.to_csv(save_dir+".csv")
        return df_t, save_dir+".csv"




if __name__ == "__main__":
    

    # log_dir = "runs/CartPole/CartPole-v1-1"

    # event_accumulator = EventAccumulator(log_dir)
    # event_accumulator.Reload()

    # events = event_accumulator.Scalars("charts/episodic_return")
    # x = [x.step for x in events]
    # y = [x.value for x in events]
    # x_interp = np.arange(1, 50000, 50)
    # y1_interp = np.interp(x_interp, x, y)

    # df = pd.DataFrame({"global_step": x_interp, "episodic_return": y1_interp})
    # df = df.assign(Method="CLAP")
    # df.to_csv("train_loss_1.csv")

    # log_dir =  "runs/CartPole/CartPole-v1-2"

    # event_accumulator = EventAccumulator(log_dir)
    # event_accumulator.Reload()

    # events = event_accumulator.Scalars("charts/episodic_return")
    # x = [x.step for x in events]
    # y = [x.value for x in events]
    # x_interp = np.arange(1, 50000, 50)
    # y1_interp = np.interp(x_interp, x, y)

    # df1 = pd.DataFrame({"global_step": x_interp, "episodic_return": y1_interp})
    # df1 = df1.assign(Method="CLAP")
    # df.to_csv("train_loss_2.csv")
    
    
    # log_dir = "runs/CartPole/CartPole-v1-3"

    # event_accumulator = EventAccumulator(log_dir)
    # event_accumulator.Reload()

    # events = event_accumulator.Scalars("charts/episodic_return")
    # x = [x.step for x in events]
    # y = [x.value for x in events]

    # x_interp = np.arange(1, 50000, 50)
    # y1_interp = np.interp(x_interp, x, y)

    # df2 = pd.DataFrame({"global_step": x_interp, "episodic_return": y1_interp})
    # df2 = df2.assign(Method="CLAP")
    # df2.to_csv("train_loss_3.csv")

    # # df_t = pd.concat([df, df1], axis=0, ignore_index=True)
    # # df_t.to_csv("train_loss.csv")
    _, path_to_csv = tb_log_to_csv(logdir_path="runs/CartPole", 
                      single_log=False, 
                      smooth=(True, 5),
                      interp_step=(True, 100, 1000), 
                      step_str="global_step", 
                      value_str="episodic_return", 
                      method_str="Clap",
                      save_path=(True, "aggr", "./csv"))
    
    painter = Painter(load_csv=True, load_dir=path_to_csv)
    
    # painter = Painter(load_csv=True, load_dir='./train_loss_1.csv')
    # painter.addCsv("./train_loss_2.csv")
    # painter.addCsv("./train_loss_3.csv")

    # #painter.smoothData('CLAP', 5)
    painter.drawFigure()

