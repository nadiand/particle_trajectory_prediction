import numpy as np
import matplotlib.pyplot as plt
from skspatial.objects import Line, Circle, Sphere

from global_constants import *


def plot_detectors(ax):
    '''
    Visualizes the 5 detectors as circular around the origin with radii
    that are 1 apart, 1...5. For 2D data they are circles and for 3D data
    they are spheres.
    '''
    if DIM == 2:
        ax.set_xlim(-6,6)
        ax.set_ylim(-6,6)
        for det_radius in np.arange(1, NR_DETECTORS+1):
            circle = Circle([0,0], det_radius)
            circle.plot_2d(ax, fill=False)
    else: # dim==3
        for det_radius in np.arange(1, NR_DETECTORS+1):
            sphere = Sphere([0,0,0], det_radius)
            sphere.plot_3d(ax, alpha=0.1)


def visualize_hits(hits_df):
    '''
    Visualizes the simplified setup: the 5 detectors as circles/spheres, depending
    on *DIM* and 5 events as the hits of the generated particles with the detectors.
    '''
    # Plot the detectors
    fig = plt.figure()
    ax = fig.add_subplot() if DIM == 2 else fig.add_subplot(projection='3d')
    plot_detectors(ax)

    # Plot 5 events
    for ind in range(5):
        row = hits_df.iloc[ind]
        for i in range(0, len(row), DIM+1):
            if DIM == 2:
                plt.plot(row[i], row[i+1], marker="o", markerfacecolor="black", markeredgecolor="black")
            else: #dim==3
                plt.plot(row[i], row[i+1], row[i+2], marker=".", markerfacecolor="black", markeredgecolor="black")
    
    plt.title("Visualization of a few events")
    plt.savefig('hits_visualized.png')
    plt.show()


def visualize_track_distribution(tracks_df):
    '''
    Creates a histogram of the trajectory parameters distribution. For the 3D
    case, both parameters are visualized in the same plot.
    '''
    tracks = []
    for i in range(0, tracks_df.shape[1], DIM):
        tracks.append(tracks_df[i].tolist())
        if DIM == 3:
            tracks.append(tracks_df[i+1].tolist())
    tracks = [param for track in tracks for param in track]

    plt.title("Distribution of tracks as given by their angles (in rad)")
    plt.hist(tracks)
    plt.show()
    plt.savefig('distr_tracks.png')


def visualize_tracks(angles, type):
    '''
    Visualizes the *type* tracks, where *type* is predicted or target, as 
    well as the detectors. The tracks are reconstructed as lines from their
    parameters (*angles*).
    '''
    
    # Plot the detectors
    fig = plt.figure()
    ax = fig.add_subplot() if DIM == 2 else fig.add_subplot(projection='3d')
    plot_detectors(ax)

    length = NR_DETECTORS+1 # TODO not sure if this is the best way, think it's not, see t1t2
    if DIM == 2:
        for angle in angles:
            x, y = np.cos(angle), np.sin(angle)
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
