import numpy as np
import pandas as pd
from skspatial.objects import Line, Circle, Sphere

from global_constants import *
from visualization import *


def intersection(radius, pt1, pt2):
    '''
    Finds and returns the 2 intersection points between a circle/sphere 
    with radius *radius* and the line connecting points *pt1* and *pt2*
    '''
    detector = Circle(pt1, radius) if DIM == 2 else Sphere(pt1, radius)
    line = Line(pt1, pt2)
    return detector.intersect_line(line)


def generation():
    '''
    Creates a dataframe of *EVENTS* many events, each with a variable 
    amount of tracks (between 1 and *MAX_NR_TRACKS*), each of which is 
    characterized by their intersection points with the detectors.
    Also creates a dataframe of the targets of these events (i.e their
    angles)
    '''
    hits = []
    tracks = []

    for _ in range(EVENTS):
        hit_entry = []
        track_entry = []
        # nr_tracks = np.random.randint(1, MAX_NR_TRACKS+1)
        nr_tracks = 3
        for t in range(nr_tracks):
            # Get a random angle for the trajectory and decide its direction
            line_direction = np.random.randint(2, size=1)[0]
            angle = np.random.uniform(-np.pi, np.pi)
            track_entry.append(angle)

            # Find the corresponding x,y(,z) coordinates
            if DIM == 2:
                x, y = np.cos(angle), np.sin(angle)
            else: # dim==3
                angle2 = np.random.uniform(-np.pi, np.pi)
                track_entry.append(angle2)
                x = np.sin(angle) * np.cos(angle2)
                y = np.sin(angle) * np.sin(angle2)
                z = np.cos(angle)
                
            # Take note of the track ID
            track_entry.append(t+1)
            
            # Find the intersection points of the detectors with the coordinates
            for ind, det_radius in enumerate(np.arange(1, NR_DETECTORS+1)):
                origin = [0,0] if DIM == 2 else [0,0,0]
                point2 = [x,y] if DIM == 2 else [x,y,z]
                # Get only the intersection points in the direction of the angle
                intersection_point = intersection(det_radius, origin, point2)[line_direction]
                for coord in intersection_point:
                    # Add noise to the hits
                    hit_entry.append(coord + np.random.normal(0, NOISE_STD, 1)[0])
                # Take note of the ID of that track of that hit
                hit_entry.append(t*NR_DETECTORS+ind+1)

        hits.append(hit_entry)
        tracks.append(track_entry)
    
    return pd.DataFrame(hits), pd.DataFrame(tracks)


if __name__ == '__main__':
    # Create data
    hits_df, tracks_df = generation()
    # Visualize a small sample as sanity check
    visualize_hits(hits_df)
    visualize_track_distribution(tracks_df)

    # Save data
    hits_df.to_csv(HITS_DATA_PATH, header=None, index=False)
    tracks_df.to_csv(TRACKS_DATA_PATH, header=None, index=False)
    print("Data generated successfully!")
