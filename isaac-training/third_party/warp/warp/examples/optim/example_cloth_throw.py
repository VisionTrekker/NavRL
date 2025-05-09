# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Grad Cloth
#
# Shows how to use Warp to optimize the initial velocities of a piece of
# cloth such that its center of mass hits a target after a specified time.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
###########################################################################

import math
import os

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


@wp.kernel
def com_kernel(positions: wp.array(dtype=wp.vec3), n: int, com: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    # compute center of mass
    wp.atomic_add(com, 0, positions[tid] / float(n))


@wp.kernel
def loss_kernel(com: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    # sq. distance to target
    delta = com[0] - target

    loss[0] = wp.dot(delta, delta)


@wp.kernel
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, stage, profile=False, verbose=False):
        self.verbose = verbose

        # seconds
        sim_duration = 2.0

        # control frequency
        self.frame_dt = 1.0 / 60.0
        frame_steps = int(sim_duration / self.frame_dt)

        # sim frequency
        self.sim_substeps = 16
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0

        self.train_iters = 64
        self.train_rate = 5.0

        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = 0.01

        dim_x = 16
        dim_y = 16

        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(0.1, 0.1, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.25),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=1.0 / dim_x,
            cell_y=1.0 / dim_y,
            mass=1.0,
            tri_ke=10000.0,
            tri_ka=10000.0,
            tri_kd=100.0,
            tri_lift=10.0,
            tri_drag=5.0,
        )

        self.profile = profile

        self.model = builder.finalize()
        self.model.ground = False

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.target = (8.0, 0.0, 0.0)
        self.com = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for i in range(self.sim_steps + 1):
            self.states.append(self.model.state(requires_grad=True))

        self.renderer = None
        if stage:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=4.0)

        # capture forward/backward passes
        self.use_graph = wp.get_device().is_cuda
        if self.use_graph:
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)
            self.graph = capture.graph

    def forward(self):
        # run control loop
        for i in range(self.sim_steps):
            self.states[i].clear_forces()

            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

        # compute loss on final state
        self.com.zero_()
        wp.launch(
            com_kernel,
            dim=self.model.particle_count,
            inputs=[self.states[-1].particle_q, self.model.particle_count, self.com],
        )
        wp.launch(loss_kernel, dim=1, inputs=[self.com, self.target, self.loss])

    def step(self):
        with wp.ScopedTimer("step", active=self.profile):
            if self.use_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)

            # gradient descent step
            x = self.states[0].particle_qd

            if self.verbose:
                print(f"Iter: {self.iter} Loss: {self.loss}")

            wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate])

            # clear grads for next iteration
            self.tape.zero()

            self.iter = self.iter + 1

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=self.profile):
            # draw trajectory
            traj_verts = [self.states[0].particle_q.numpy().mean(axis=0)]

            for i in range(0, self.sim_steps, self.sim_substeps):
                traj_verts.append(self.states[i].particle_q.numpy().mean(axis=0))

                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.states[i])
                self.renderer.render_box(
                    pos=self.target,
                    rot=wp.quat_identity(),
                    extents=(0.1, 0.1, 0.1),
                    name="target",
                    color=(1.0, 0.0, 0.0),
                )
                self.renderer.render_line_strip(
                    vertices=traj_verts,
                    color=wp.render.bourke_color_map(0.0, 269.0, self.loss.numpy()[0]),
                    radius=0.02,
                    name=f"traj_{self.iter-1}",
                )
                self.renderer.end_frame()

                self.render_time += self.frame_dt


if __name__ == "__main__":
    stage_path = os.path.join(wp.examples.get_output_directory(), "example_cloth_throw.usd")

    example = Example(stage_path, profile=False, verbose=True)

    # replay and optimize
    for i in range(example.train_iters):
        example.step()
        if i % 4 == 0:
            example.render()

    if example.renderer:
        example.renderer.save()
