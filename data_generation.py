import numpy as np
import pandas as pd
from skspatial.objects import Line, Circle, Sphere
from global_constants import *
from visualization import *


def intersection(radius, pt1, pt2):
    detector = Circle(pt1, radius) if DIM == 2 else Sphere(pt1, radius)
    line = Line(pt1, pt2)
    return detector.intersect_line(line)

def generation():
    data = []
    track_params = []

    for id in range(EVENTS):
        row = []
        param_row = []
        nr_tracks = np.random.randint(2, MAX_NR_TRACKS+1)
        for t in range(nr_tracks):
            line_direction = np.random.randint(2, size=1)[0]
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
            
            for ind, d in enumerate(np.arange(1, NR_DETECTORS+1)):
                origin = [0,0] if DIM == 2 else [0,0,0]
                point2 = [x,y] if DIM == 2 else [x,y,z]
                intersection_point = intersection(d, origin, point2)[line_direction]
                for coord in intersection_point:
                    row.append(coord + np.random.normal(0, NOISE_STD, 1)[0])
                row.append(t*NR_DETECTORS+ind+1) # hit1trackid1

        data.append(row)
        track_params.append(param_row)
    
    return pd.DataFrame(data), pd.DataFrame(track_params)

if __name__ == '__main__':
    hits_df, tracks_df = generation()
    visualize_hits(hits_df)
    visualize_track_distribution(tracks_df)
    hits_df.to_csv(HITS_DATA_PATH, header=None, index=False)
    tracks_df.to_csv(TRACKS_DATA_PATH, header=None, index=False)
    print("Data generated successfully!")
