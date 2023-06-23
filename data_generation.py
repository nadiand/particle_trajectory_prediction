import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from global_constants import *

from skspatial.objects import Line, Circle, Sphere

def circle_intersection(radius, pt1, pt2):
    circle = Circle([0,0], radius)
    line = Line(pt1, pt2)
    intersection_point, _ = circle.intersect_line(line)
    return intersection_point # of the form [x,y]

def sphere_intersection(radius, pt1, pt2):
    sphere = Sphere([0,0,0], radius)
    line = Line(pt1, pt2)
    intersection_point, _ = sphere.intersect_line(line)
    return intersection_point # of the form [x,y,z]

def visualize(hits_df, tracks_df):
    ax = plt.gca()
    for d in range(NR_DETECTORS+1):
        circle = plt.Circle((0,0), d, alpha=0.5, fill=False)
        ax.add_artist(circle)
        ax.set_ylim(-7,7)
        ax.set_xlim(-7,7)

    for i in range(5):
        row = hits_df.iloc[i]
        for i in range(0, len(row), 3):
            plt.plot(row[i], row[i + 1], marker="o", markerfacecolor="black", markeredgecolor="black")

    plt.title("Visualization of a few events")
    plt.savefig('hits_visualized.png')
    plt.show()

    plt.title("Distribution of tracks as given by their angles (in rad)")
    plt.hist(tracks_df[0].append(tracks_df[2]).append(tracks_df[4]))
    plt.show()
    plt.savefig('distr_tracks.png')

if __name__ == '__main__':
    data = []
    track_params = []

    for id in range(EVENTS):
        row = []
        param_row = []
        for t in range(TRACKS_PER_EVENT):
            angle = np.random.uniform(-np.pi, np.pi)
            param_row.append(angle) # parameter of the hit, its angle
            param_row.append(t+1) # id of the track
            x, y = np.cos(angle), np.sin(angle)
            for ind, d in enumerate(np.arange(1, NR_DETECTORS+1)):
                intersection = circle_intersection(d, [0,0], [x,y])
                row.append(intersection[0] + np.random.normal(0, NOISE_STD, 1)[0])
                row.append(intersection[1] + np.random.normal(0, NOISE_STD, 1)[0])
                row.append(t*NR_DETECTORS+ind+1)
        data.append(row)
        track_params.append(param_row)
    
    hits_df = pd.DataFrame(data)
    tracks_df = pd.DataFrame(track_params)
    visualize(hits_df, tracks_df)
    hits_df.to_csv(HITS_DATA_PATH, header=None)
    tracks_df.to_csv(TRACKS_DATA_PATH, header=None)
    print("Data generated successfully!")
