import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_array(arr,filename, show, save):
    
    fig = plt.figure()

    
    ims = []
    for i in range(arr.shape[0]):
        im = plt.imshow(arr[i],animated=True)
        plt.title('title')
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=10000)

    if show == True:
        plt.show()

    if save == True:
         ani.save(filename + '.mp4')
         
with h5py.File('/global/cscratch1/sd/jpathak/rbc2d/data/snapshots_Ra7Pr0xres512zres512seed4_s3.h5', 'r') as f:
    
    animate_array(f['tasks']['b'], 'rbc512', True, True)
    