import numpy as np

import warp as wp

import os

wp.init()


class Example:

    def __init__(self, stage):

        self.device = wp.get_preferred_device()
        self.grid_cell_size = self.point_radius*5.0
        dimX=100
        dimY=100
        dimZ=100
        self.grid = wp.HashGrid(128, 128, 128, self.device)

        self.points = self.particle_grid(32, 128, 32, (0.0, 0.3, 0.0), self.point_radius, 0.1)

        self.x = wp.array(self.points, dtype=wp.vec3, device=self.device)
        self.v = wp.array(np.ones([len(self.x), 3])*np.array([0.0, 0.0, 10.0]), dtype=wp.vec3, device=self.device)
        self.f = wp.zeros_like(self.v)

        self.use_graph = (self.device == "cuda")

        if (self.use_graph):

            wp.capture_begin()

            for s in range(self.sim_substeps):

                with wp.ScopedTimer("forces", active=False):
                    wp.launch(kernel=apply_forces, dim=len(self.x), inputs=[self.grid.id, self.x, self.v, self.f, self.point_radius, self.k_contact, self.k_damp, self.k_friction, self.k_mu], device=self.device)
                    wp.launch(kernel=integrate, dim=len(self.x), inputs=[self.x, self.v, self.f, (0.0, -9.8, 0.0), self.sim_dt, self.inv_mass], device=self.device)
                
            self.graph = wp.capture_end()

    def update(self):

        with wp.ScopedTimer("simulate", active=True):

            if (self.use_graph):

                with wp.ScopedTimer("grid build", active=False):
                    self.grid.build(self.x, self.grid_cell_size)

                with wp.ScopedTimer("solve", active=False):
                    wp.capture_launch(self.graph)
                    wp.synchronize()

            else:
                for s in range(self.sim_substeps):

                    with wp.ScopedTimer("grid build", active=False):
                        self.grid.build(self.x, self.point_radius)

                    with wp.ScopedTimer("forces", active=False):
                        wp.launch(kernel=apply_forces, dim=len(self.x), inputs=[self.grid.id, self.x, self.v, self.f, self.point_radius, self.k_contact, self.k_damp, self.k_friction, self.k_mu], device=self.device)
                        wp.launch(kernel=integrate, dim=len(self.x), inputs=[self.x, self.v, self.f, (0.0, -9.8, 0.0), self.sim_dt, self.inv_mass], device=self.device)
                
                wp.synchronize()

    def render(self, is_live=False):

        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time 
            
            self.renderer.begin_frame(time)
            self.renderer.render_points(points=self.x.numpy(), radius=self.point_radius, name="points")
            self.renderer.end_frame()

        self.sim_time += self.frame_dt

    # creates a grid of particles
    def particle_grid(self, dim_x, dim_y, dim_z, lower, radius, jitter):
        points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
        points_t = np.array((points[0], points[1], points[2])).T*radius*2.0 + np.array(lower)
        points_t = points_t + np.random.rand(*points_t.shape)*radius*jitter
        
        return points_t.reshape((-1, 3))