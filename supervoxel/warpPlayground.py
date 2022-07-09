


import numpy as np

import warp as wp
from warp.tests.test_base import *

np.random.seed(532)


num_points = 4096
dim_x = 128
dim_y = 128
dim_z = 128

scale = 150.0

cell_radius = 8.0
query_radius = 8.0

num_runs = 4

print_enabled = False

@wp.kernel
def count_neighbors(grid : wp.uint64,
                    radius: float,
                    points: wp.array(dtype=wp.vec3),
                    counts: wp.array(dtype=int)):

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # query point    
    p = points[i]
    count = int(0)

    # construct query around point p
    neighbors = wp.hash_grid_query(grid, p, radius)

    for index in neighbors:

        # compute distance to point
        d = wp.length(p - points[index])

        if (d <= radius):
            count += 1

    counts[i] = count

print("aslkjd")