import sys
import numpy as np
import argparse
# import tkinter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def plot3D(ax,stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = np.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    z = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
            z.append(traj[i][2])
        last= stamps[i]
    if len(x)>0:

        # ax = fig.add_subplot(111, projection='3d')
        print(len(x))
        print(len(y))
        ax.plot(x, y, zs=z)#,color=color,label=label)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("3D Plot")
        plt.grid(True)        
        # ax.plot(x,y,style,color=color,label=label)
        # plt.show(block = False)

        # input("Pausing")
