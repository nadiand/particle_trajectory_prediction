import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from global_constants import *

# from https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py
def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections
        
def add_noise(value):
  noise = np.random.normal(0, NOISE_STD, 1)[0]
  return value + noise

def plot_circle(center, radius, color='blue', alpha=0.5, ax=None):
  if ax is None:
      ax = plt.gca()
  circle = plt.Circle(center, radius, color=color, alpha=alpha, fill=False)
  ax.add_artist(circle)
  ax.set_ylim(-7,7)
  ax.set_xlim(-7,7)

def visualize(hits_df, tracks_df):
    for d in range(NR_DETECTORS):
        plot_circle((0,0), d)

    for i in range(5):
        row = hits_df.iloc[i]
        for i in range(0, len(row), 3):
            plt.plot(row[i], row[i + 1], marker="o", markerfacecolor="black", markeredgecolor="black")

    plt.title("Visualization of a few events")
    plt.savefig('hits_visualized.png')

    plt.title("Distribution of tracks as given by their angles (in rad)")
    plt.hist(tracks_df[0].append(tracks_df[2]).append(tracks_df[4]))
    plt.savefig('distr_tracks.png')

if __name__ == '__main__':
    data = []
    track_params = []

    for id in range(EVENTS):
        row = []
        param_row = []
        for t in range(TRACKS_PER_EVENT):
            angle = np.random.uniform(-np.pi, np.pi)
            param_row.append(angle)
            param_row.append(t+1)
            x, y = np.cos(angle), np.sin(angle)
            for ind, d in enumerate(np.arange(1, NR_DETECTORS)):
                intersection = circle_line_segment_intersection((0,0), d, (0,0), (x,y))[1]
                row.append(add_noise(intersection[0]))
                row.append(add_noise(intersection[1]))
                row.append(t*NR_DETECTORS+ind+1)
        data.append(row)
        track_params.append(param_row)
    
    hits_df = pd.DataFrame(data)
    tracks_df = pd.DataFrame(track_params)
    # visualize(hits_df, tracks_df)
    hits_df.to_csv(HITS_DATA_PATH, index=False)
    tracks_df.to_csv(TRACKS_DATA_PATH, index=False)
    print("Data generated successfully!")