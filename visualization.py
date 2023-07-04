import numpy as np
import matplotlib.pyplot as plt
from skspatial.objects import Line, Circle, Sphere

from global_constants import *


def visualize_hits(hits_df):
    if DIM == 2:
        fig, ax = plt.subplots()
        ax.set_ylim(-6,6)
        ax.set_xlim(-6,6)
        for d in np.arange(1, NR_DETECTORS+1):
            circle = Circle([0,0], d)
            circle.plot_2d(ax, fill=False)
    else: # dim==3
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for d in np.arange(1, NR_DETECTORS+1):
            sphere = Sphere([0,0,0], d)
            sphere.plot_3d(ax, alpha=0.1)

    for ind in range(5):
        row = hits_df.iloc[ind]
        for i in range(0, len(row), DIM+1):
            if DIM == 2:
                plt.plot(row[i], row[i+1], marker="o", markerfacecolor="black", markeredgecolor="black")
            else: 
                plt.plot(row[i], row[i+1], row[i+2], marker="o", markerfacecolor="black", markeredgecolor="black")
    
    plt.title("Visualization of a few events")
    plt.savefig('hits_visualized.png')
    plt.show()


def visualize_track_distribution(tracks_df):
    # Same for 2D and 3D data
    plt.title("Distribution of tracks as given by their angles (in rad)")
    tracks = []
    for i in range(0, tracks_df.shape[1], DIM):
        tracks.append(tracks_df[i].tolist())
        if DIM == 3:
            tracks.append(tracks_df[i+1].tolist())
    tracks = [item for sublist in tracks for item in sublist]
    plt.hist(tracks)
    plt.show()
    plt.savefig('distr_tracks.png')


def visualize_tracks(angles, type):
    length = NR_DETECTORS+1
    
    if DIM == 2:
        fig, ax = plt.subplots()
        ax.set_ylim(-6,6)
        ax.set_xlim(-6,6)
        for d in np.arange(1, NR_DETECTORS+1):
            circle = Circle([0,0], d)
            circle.plot_2d(ax, fill=False)
    else: # dim==3
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for d in np.arange(1, NR_DETECTORS+1):
            sphere = Sphere([0,0,0], d)
            sphere.plot_3d(ax, alpha=0.1)

    if DIM == 2:
        for angle in angles:
            x, y = np.cos(angle), np.sin(angle)
            print(x,y)
            x = x+length if x > 0 else x-length
            y = y+length if y > 0 else y-length
            #TODO try with the t1,t2 parameters of line ! 
            # https://scikit-spatial.readthedocs.io/en/stable/api_reference/Line/methods/skspatial.objects.Line.plot_2d.html#skspatial.objects.Line.plot_2d
            line = Line([0,0], [x,y])
            line.plot_2d(ax)
    else: #dim==3
        for angle, angle2 in angles:
            x, y, z = np.sin(angle) * np.cos(angle2), np.sin(angle) * np.sin(angle2), np.cos(angle)
            line = Line([0,0,0], [x,y,z])
            line.plot_3d(ax)
        
    plt.title(f"Reconstruction of {type} tracks")
    plt.show()
