import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skspatial.objects import Line, Circle, Sphere
from global_constants import *


def intersection(radius, pt1, pt2):
    detector = Circle(pt1, radius) if DIM == 2 else Sphere(pt1, radius)
    line = Line(pt1, pt2)
    intersection_point, _ = detector.intersect_line(line)
    return intersection_point


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


if __name__ == '__main__':
    data = []
    track_params = []

    for id in range(EVENTS):
        row = []
        param_row = []
        nr_tracks = np.random.randint(2, MAX_NR_TRACKS+1)
        for t in range(nr_tracks):
            angle = np.random.uniform(-np.pi, np.pi)
            param_row.append(angle) # parameter of the hit, its angle

            if DIM == 2:
                x, y = np.cos(angle), np.sin(angle)
            else: # dim==3
                angle2 = np.random.uniform(-np.pi, np.pi)
                param_row.append(angle2)
                x = np.sin(angle) * np.cos(angle2)
                y = np.sin(angle) * np.sin(angle2)
                z = np.cos(angle)
                
            param_row.append(t+1) # id of the track

            # do i have to do 5 * the coords to get a long enough line? TODO
            
            for ind, d in enumerate(np.arange(1, NR_DETECTORS+1)):
                origin = [0,0] if DIM == 2 else [0,0,0]
                point2 = [x,y] if DIM == 2 else [x,y,z]
                intersection_point = intersection(d, origin, point2) 
                for coord in intersection_point:
                    row.append(coord + np.random.normal(0, NOISE_STD, 1)[0])
                row.append(t*NR_DETECTORS+ind+1) # hit1trackid1

        data.append(row)
        track_params.append(param_row)
    
    hits_df = pd.DataFrame(data)
    tracks_df = pd.DataFrame(track_params)
    visualize_hits(hits_df)
    # visualize_track_distribution(tracks_df)
    hits_df.to_csv(HITS_DATA_PATH, header=None, index=False)
    tracks_df.to_csv(TRACKS_DATA_PATH, header=None, index=False)
    print("Data generated successfully!")
