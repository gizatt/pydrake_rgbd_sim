# -*- coding: utf8 -*-

import argparse
import os
import time

import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import pydrake
from pydrake.all import (
    AbstractValue,
    AddFlatTerrainToWorld,
    AddModelInstancesFromSdfString,
    AddModelInstanceFromUrdfFile,
    DiagramBuilder,
    FloatingBaseType,
    Image,
    LeafSystem,
    PixelType,
    PortDataType,
    RgbdCamera,
    RigidBodyFrame,
    RigidBodyPlant,
    RigidBodyTree,
    RungeKutta2Integrator,
    Shape,
    SignalLogger,
    Simulator,
)

from underactuated.meshcat_rigid_body_visualizer import (
    MeshcatRigidBodyVisualizer)

import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g


def setup_tabletop(rbt):
    table_sdf_path = os.path.join(
        pydrake.getDrakePath(),
        "examples", "kuka_iiwa_arm", "models", "table",
        "extra_heavy_duty_table_surface_only_collision.sdf")

    object_urdf_path = os.path.join(
        pydrake.getDrakePath(),
        "examples", "kuka_iiwa_arm", "models", "objects",
        "block_for_pick_and_place.urdf")

    AddFlatTerrainToWorld(rbt)
    table_frame_robot = RigidBodyFrame(
        "table_frame_robot", rbt.world(),
        [0.0, 0, 0], [0, 0, 0])
    AddModelInstancesFromSdfString(
        open(table_sdf_path).read(), FloatingBaseType.kFixed,
        table_frame_robot, rbt)

    table_top_z_in_world = 0.736 + 0.057 / 2

    object_init_frame = RigidBodyFrame(
        "object_init_frame", rbt.world(),
        [0.0, 0, table_top_z_in_world+0.1],
        [0, 0, 0])
    AddModelInstanceFromUrdfFile(object_urdf_path,
                                 FloatingBaseType.kRollPitchYaw,
                                 object_init_frame, rbt)

    np.random.seed(42)
    for i in range(5):
        hp = (np.random.random(2)-0.5)*0.5
        object_init_frame = RigidBodyFrame(
            "object_init_frame", rbt.world(),
            [hp[0], hp[1], table_top_z_in_world+0.15],
            np.random.random(3))
        AddModelInstanceFromUrdfFile(object_urdf_path,
                                     FloatingBaseType.kRollPitchYaw,
                                     object_init_frame, rbt)


class DepthImageCorruptionBlock(LeafSystem):
    def __init__(self,
                 camera):
        LeafSystem.__init__(self)
        self.set_name('depth image corruption superclass')
        self.camera = camera

        self.depth_image_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued,
                                   camera.depth_image_output_port().size())

        self.depth_image_output_port = \
            self._DeclareAbstractOutputPort(
                self._DoAllocDepthCameraImage,
                self._DoCalcAbstractOutput)

    def _DoAllocDepthCameraImage(self):
        test = AbstractValue.Make(Image[PixelType.kDepth32F](
            self.camera.depth_camera_info().width(),
            self.camera.depth_camera_info().height()))
        print "Allocated ", test
        return test

    def _DoCalcAbstractOutput(self, context, y_data):
        print "OVERRIDE ME"
        sys.exit(-1)


class DepthImageHeuristicCorruptionBlock(DepthImageCorruptionBlock):
    def __init__(self,
                 camera):
        DepthImageCorruptionBlock.__init__(self, camera)
        self.set_name('depth image corruption, heuristic')

        self.rgbd_normal_limit = 0.5
        self.rgbd_noise = 0.005
        self.rgbd_projector_baseline = 0.2
        self.near_distance = 0.5
        self.far_distance = 2.0

        # Cache these things that are used in every loop
        # to minimize re-allocation of these big arrays
        K = self.camera.depth_camera_info().intrinsic_matrix()
        w = self.camera.depth_camera_info().width()
        h = self.camera.depth_camera_info().height()
        # How much does each depth point project laterally
        # (in the axis of the camera-projector pair?)
        self.x_indices_im = np.tile(np.arange(w), [h, 1])

    def _DoCalcAbstractOutput(self, context, y_data):
        u_data = self.EvalAbstractInput(context, 0).get_value()
        h, w, _ = u_data.data.shape
        depth_image = np.empty((h, w), dtype=np.float32)
        depth_image[:, :] = u_data.data[:, :, 0]
        good_mask = np.isfinite(depth_image)
        depth_image = np.clip(depth_image, self.near_distance,
                              self.far_distance)

        # Calculate normals before adding noise
        if self.rgbd_normal_limit > 0.:
            gtNormalImage = np.absolute(
                cv2.Scharr(depth_image, cv2.CV_32F, 1, 0)) + \
                np.absolute(cv2.Scharr(depth_image, cv2.CV_32F, 0, 1))
            _, normalThresh = cv2.threshold(
                gtNormalImage, self.rgbd_normal_limit,
                1., cv2.THRESH_BINARY_INV)

        if self.rgbd_noise > 0.0:
            noiseMat = np.random.randn(h, w)*self.rgbd_noise
            depth_image += noiseMat

        depth_image_out = np.zeros(depth_image.shape)
        depth_image_out += depth_image

        if self.rgbd_projector_baseline > 0.0:

            K = self.camera.depth_camera_info().intrinsic_matrix()
            x_projection = (self.x_indices_im - K[0, 2]) * \
                depth_image / K[0, 0]

            # For a fixed shift...
            for shift_amt in range(-50, 0, 1):
                imshift_tf_matrix = np.array(
                    [[1., 0., shift_amt], [0., 1., 0.]])
                sh_x_projection = cv2.warpAffine(
                    x_projection, imshift_tf_matrix,
                    (w, h), borderMode=cv2.BORDER_REPLICATE)
                shifted_gt_depth = cv2.warpAffine(
                    depth_image, imshift_tf_matrix, (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=float(np.max(depth_image)))

                # (projected test point - projected original point) dot
                # producted with vector perpendicular to sample point
                # and projector origin
                error_im = (sh_x_projection - x_projection)*(-depth_image) + \
                    (shifted_gt_depth - depth_image) * \
                    (x_projection - self.rgbd_projector_baseline)

                # TODO, fix this hard-convert-back-to-32bit-float
                # Threshold any positive error as occluded
                _, error_thresh = cv2.threshold(
                    error_im.astype(np.float32), 0., 1.,
                    cv2.THRESH_BINARY_INV)
                depth_image_out *= error_thresh

        # Apply normal limiting
        if self.rgbd_normal_limit > 0.:
            depth_image_out *= normalThresh

        # Where it's infinite, set to 0
        depth_image_out = np.where(
            good_mask, depth_image_out,
            np.zeros(depth_image.shape))
        y_data.get_mutable_value().mutable_data[:, :, 0] = \
            depth_image_out[:, :]


class RgbdCameraMeshcatVisualizer(LeafSystem):
    def __init__(self,
                 camera,
                 rbt,
                 draw_timestep=0.033333,
                 prefix="RBCameraViz",
                 zmq_url="tcp://127.0.0.1:6000"):
        LeafSystem.__init__(self)
        self.set_name('camera meshcat visualization')
        self.timestep = draw_timestep
        self._DeclarePeriodicPublish(draw_timestep, 0.0)
        self.camera = camera
        self.rbt = rbt
        self.prefix = prefix

        self.camera_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued,
                                   camera.depth_image_output_port().size())
        self.state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt.get_num_positions() +
                                   rbt.get_num_velocities())

        self.ax = None

        # Set up meshcat
        self.vis = meshcat.Visualizer(zmq_url=zmq_url)
        self.vis[prefix].delete()

    def _DoPublish(self, context, event):
        u_data = self.EvalAbstractInput(context, 0).get_value()
        x = self.EvalVectorInput(context, 1).get_value()
        w, h, _ = u_data.data.shape
        depth_image = u_data.data[:, :, 0]

        if self.ax is None:
            self.ax = plt.imshow(depth_image)
        else:
            self.ax.set_data(depth_image)
        plt.pause(1E-12)

        # Convert depth image to point cloud, with +z being
        # camera "forward"
        Kinv = np.linalg.inv(
            self.camera.depth_camera_info().intrinsic_matrix())
        U, V = np.meshgrid(np.arange(h), np.arange(w))
        points_in_camera_frame = np.vstack([
            U.flatten(),
            V.flatten(),
            np.ones(w*h)])
        points_in_camera_frame = Kinv.dot(points_in_camera_frame) * \
            depth_image.flatten()

        # The depth camera has some offset from the camera's root frame,
        # so take than into account.
        pose_mat = self.camera.depth_camera_optical_pose().matrix()
        points_in_camera_frame = pose_mat[0:3, 0:3].dot(points_in_camera_frame)
        points_in_camera_frame += np.tile(pose_mat[0:3, 3], [w*h, 1]).T

        kinsol = self.rbt.doKinematics(x)
        points_in_world_frame = self.rbt.transformPoints(
            kinsol,
            points_in_camera_frame,
            self.camera.frame().get_frame_index(),
            0)

        # Color points according to their normalized height
        min_height = 0.7
        max_height = 1.0
        colors = cm.jet(
            (points_in_world_frame[2, :]-min_height)/(max_height-min_height)
            ).T[0:3, :]

        self.vis[self.prefix]["points"].set_object(
            g.PointCloud(position=points_in_world_frame,
                         color=colors,
                         size=0.005))


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", "--duration",
                        type=float,
                        help="Duration to run sim.",
                        default=1000.0)
    parser.add_argument("--test",
                        action="store_true",
                        help="Help out CI by launching a meshcat server for "
                             "the duration of the test.")
    args = parser.parse_args()

    meshcat_server_p = None
    if args.test:
        print "Spawning"
        import subprocess
        meshcat_server_p = subprocess.Popen(["meshcat-server"])
    else:
        print "Warning: if you have not yet run meshcat-server in another " \
              "terminal, this will hang."

    # Construct the robot and its environment
    rbt = RigidBodyTree()
    setup_tabletop(rbt)

    # Set up a visualizer for the robot
    pbrv = MeshcatRigidBodyVisualizer(rbt, draw_timestep=0.01)
    # (wait while the visualizer warms up and loads in the models)
    time.sleep(2.0)

    # Plan a robot motion to maneuver from the initial posture
    # to a posture that we know should grab the object.
    # (Grasp planning is left as an exercise :))
    q0 = rbt.getZeroConfiguration()

    # Make our RBT into a plant for simulation
    rbplant = RigidBodyPlant(rbt)
    rbplant.set_name("Rigid Body Plant")

    # Build up our simulation by spawning controllers and loggers
    # and connecting them to our plant.
    builder = DiagramBuilder()
    # The diagram takes ownership of all systems
    # placed into it.
    rbplant_sys = builder.AddSystem(rbplant)

    # Hook up the visualizer we created earlier.
    visualizer = builder.AddSystem(pbrv)
    builder.Connect(rbplant_sys.state_output_port(),
                    visualizer.get_input_port(0))

    # Add a camera, too, though no controller or estimator
    # will consume the output of it.
    # - Add frame for camera fixture.
    camera_frame = RigidBodyFrame(
        name="rgbd camera frame", body=rbt.world(),
        xyz=[1.0, 0., 1.25], rpy=[0., 0.5, -np.pi])
    rbt.addFrame(camera_frame)
    camera = builder.AddSystem(
        RgbdCamera(name="camera", tree=rbt, frame=camera_frame,
                   z_near=0.5, z_far=2.0, fov_y=np.pi / 4,
                   width=640, height=480,
                   show_window=False))
    builder.Connect(rbplant_sys.state_output_port(),
                    camera.get_input_port(0))

    depth_corruptor = builder.AddSystem(
        DepthImageHeuristicCorruptionBlock(camera))
    builder.Connect(camera.depth_image_output_port(),
                    depth_corruptor.depth_image_input_port)

    camera_meshcat_visualizer = builder.AddSystem(
        RgbdCameraMeshcatVisualizer(camera, rbt))

    builder.Connect(depth_corruptor.depth_image_output_port,
                    camera_meshcat_visualizer.camera_input_port)
    builder.Connect(rbplant_sys.state_output_port(),
                    camera_meshcat_visualizer.state_input_port)

    # Done!
    diagram = builder.Build()

    # Create a simulator for it.
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    # Simulator time steps will be very small, so don't
    # force the rest of the system to update every single time.
    simulator.set_publish_every_time_step(False)

    # The simulator simulates forward from a given Context,
    # so we adjust the simulator's initial Context to set up
    # the initial state.
    state = simulator.get_mutable_context().\
        get_mutable_continuous_state_vector()
    initial_state = np.zeros(state.size())
    initial_state[0:q0.shape[0]] = q0
    state.SetFromVector(initial_state)

    # From iiwa_wsg_simulation.cc:
    # When using the default RK3 integrator, the simulation stops
    # advancing once the gripper grasps the box.  Grasping makes the
    # problem computationally stiff, which brings the default RK3
    # integrator to its knees.
    timestep = 0.0001
    simulator.reset_integrator(
        RungeKutta2Integrator(diagram, timestep,
                              simulator.get_mutable_context()))

    # This kicks off simulation. Most of the run time will be spent
    # in this call.
    simulator.StepTo(args.duration)
    print("Final state: ", state.CopyToVector())

    if meshcat_server_p is not None:
        meshcat_server_p.kill()
        meshcat_server_p.wait()
