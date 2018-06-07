# -*- coding: utf8 -*-

import argparse
import os
import random
import time

import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import pydrake
from pydrake.solvers import ik
from pydrake.all import (
    AbstractValue,
    AddFlatTerrainToWorld,
    AddModelInstancesFromSdfString,
    AddModelInstanceFromUrdfFile,
    CompliantMaterial,
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

from RGBDCNN import network


def save_image_uint16(name, im):
    array_as_uint16 = im.astype(np.uint16)
    sp.misc.imsave(name, im)


def save_image_uint8(name, im):
    array_as_uint8 = im.astype(np.uint8)
    sp.misc.imsave(name, im)


def save_image_colormap(name, im):
    plt.imsave(name, im, cmap=plt.cm.inferno)


def save_depth_colormap(name, im, near, far):
    cmapped = plt.cm.jet((far - im)/(far - near))
    zero_range_mask = im < near
    cmapped[:, :, 0][zero_range_mask] = 0.0
    cmapped[:, :, 1][zero_range_mask] = 0.0
    cmapped[:, :, 2][zero_range_mask] = 0.0
    sp.misc.imsave(name, cmapped)


def setup_tabletop(rbt):
    table_sdf_path = os.path.join(
        pydrake.getDrakePath(),
        "examples", "kuka_iiwa_arm", "models", "table",
        "extra_heavy_duty_table_surface_only_collision.sdf")

    object_urdf_paths = [
        os.path.join(
            pydrake.getDrakePath(),
            "examples", "kuka_iiwa_arm", "models", "objects",
            "block_for_pick_and_place.urdf"),
        os.path.join(
            pydrake.getDrakePath(),
            "examples", "kuka_iiwa_arm", "models", "objects",
            "big_robot_toy.urdf"),
        os.path.join(
            pydrake.getDrakePath(),
            "examples", "kuka_iiwa_arm", "models", "objects",
            "simple_cylinder.urdf"),
        os.path.join(
            "models", "rlg_misc_models", "companion_cube.urdf"),
        os.path.join(
            "models", "rlg_misc_models", "apriltag_cube.urdf"),
        os.path.join(
            "models", "dish_models", "bowl_6p25in.urdf"),
        os.path.join(
            "models", "dish_models", "bowl_6p25in.urdf"),
        os.path.join(
            "models", "dish_models", "plate_11in.urdf"),
        ]

    AddFlatTerrainToWorld(rbt)
    table_frame_robot = RigidBodyFrame(
        "table_frame_robot", rbt.world(),
        [0.0, 0, 0], [0, 0, 0])
    AddModelInstancesFromSdfString(
        open(table_sdf_path).read(), FloatingBaseType.kFixed,
        table_frame_robot, rbt)

    table_top_z_in_world = 0.736 + 0.057 / 2

    for i in range(len(object_urdf_paths)):
        hp = (np.random.random(2)-0.5)*0.25
        object_init_frame = RigidBodyFrame(
            "object_init_frame", rbt.world(),
            [hp[0], hp[1], table_top_z_in_world+0.05],
            np.random.random(3))
        AddModelInstanceFromUrdfFile(
            object_urdf_paths[i % len(object_urdf_paths)],
            FloatingBaseType.kRollPitchYaw,
            object_init_frame, rbt)

    rbt.compile()

    # Project arrangement to nonpenetration with IK
    constraints = []

    constraints.append(ik.MinDistanceConstraint(
        model=rbt, min_distance=0.01, active_bodies_idx=list(),
        active_group_names=set()))

    q0 = np.zeros(rbt.get_num_positions())
    options = ik.IKoptions(rbt)
    options.setDebug(True)
    options.setMajorIterationsLimit(10000)
    options.setIterationsLimit(100000)
    results = ik.InverseKin(
        rbt, q0, q0, constraints, options)

    qf = results.q_sol[0]
    info = results.info[0]
    print "Projected with info %d" % info
    return qf


class DepthImageCorruptionBlock(LeafSystem):
    def __init__(self, camera, save_dir):
        LeafSystem.__init__(self)
        self.set_name('depth image corruption superclass')
        self.camera = camera

        self.save_dir = save_dir
        if self.save_dir is not None:
            os.system("rm -r %s" % self.save_dir)
            os.system("mkdir -p %s" % self.save_dir)

        self.depth_image_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued,
                                   camera.depth_image_output_port().size())

        self.color_image_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued,
                                   camera.color_image_output_port().size())

        self.depth_image_output_port = \
            self._DeclareAbstractOutputPort(
                self._DoAllocDepthCameraImage,
                self._DoCalcAbstractOutput)

    def _DoAllocDepthCameraImage(self):
        test = AbstractValue.Make(Image[PixelType.kDepth32F](
            self.camera.depth_camera_info().width(),
            self.camera.depth_camera_info().height()))
        return test

    def _DoCalcAbstractOutput(self, context, y_data):
        print "OVERRIDE ME"
        sys.exit(-1)


class DepthImageHeuristicCorruptionBlock(DepthImageCorruptionBlock):
    def __init__(self, camera, save_dir):
        DepthImageCorruptionBlock.__init__(self, camera, save_dir)
        self.set_name('depth image corruption, heuristic')

        self.rgbd_normal_limit = 0.0
        self.rgbd_noise = 0.000
        self.rgbd_projector_baseline = 0.05
        self.rgbd_rectification_baseline = -0.025
        self.near_distance = 0.2
        self.far_distance = 3.5

        # Cache these things that are used in every loop
        # to minimize re-allocation of these big arrays
        K = self.camera.depth_camera_info().intrinsic_matrix()
        K_rgb = self.camera.color_camera_info().intrinsic_matrix()
        print "POSES: ",
        print self.camera.color_camera_optical_pose()
        print self.camera.depth_camera_optical_pose()
        w = self.camera.depth_camera_info().width()
        h = self.camera.depth_camera_info().height()
        # How much does each depth point project laterally
        # (in the axis of the camera-projector pair?)
        x_inds, y_inds = np.meshgrid(np.arange(w), np.arange(h))
        self.xy1_indices_im = np.dstack([
            x_inds, y_inds, np.ones((h, w))])
        self.iter = 0

    def _DoCalcAbstractOutput(self, context, y_data):
        start_time = time.time()

        u_data = self.EvalAbstractInput(context, 1).get_value()
        h, w, _ = u_data.data.shape
        rgb_image = np.empty((h, w), dtype=np.float64)
        rgb_image[:, :] = u_data.data[:, :, 0]

        if self.save_dir is not None:
            save_image_uint8(
                "%s/%05d_rgb.png" % (self.save_dir, self.iter), rgb_image)

        u_data = self.EvalAbstractInput(context, 0).get_value()
        h, w, _ = u_data.data.shape
        depth_image = np.empty((h, w), dtype=np.float32)
        depth_image[:, :] = u_data.data[:, :, 0]
        good_mask = np.isfinite(depth_image)
        depth_image = np.clip(depth_image, self.near_distance,
                              self.far_distance)

        if self.save_dir is not None:
            save_depth_colormap(
                "%s/%05d_input_depth.png" % (self.save_dir, self.iter),
                depth_image, self.near_distance, self.far_distance)

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

        if self.rgbd_projector_baseline > 0.0:

            K = self.camera.depth_camera_info().intrinsic_matrix()
            x_projection = (self.xy1_indices_im[:, :, 0] - K[0, 2]) * \
                depth_image / K[0, 0]

            # For a fixed shift...
            mask = np.ones(depth_image.shape)
            for shift_amt in range(-50, 0, 10):
                imshift_tf_matrix = np.array(
                    [[1., 0., shift_amt], [0., 1., 0.]])
                sh_x_projection = cv2.warpAffine(
                    x_projection, imshift_tf_matrix,
                    (w, h), borderMode=cv2.BORDER_REPLICATE)
                shifted_gt_depth = cv2.warpAffine(
                    depth_image, imshift_tf_matrix, (w, h),
                    borderMode=cv2.BORDER_REPLICATE)

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
                mask *= error_thresh
            depth_image *= mask

            if self.save_dir is not None:
                save_image_colormap(
                    "%s/%05d_mask.png" % (self.save_dir, self.iter), mask)
                save_depth_colormap(
                    "%s/%05d_prerectified_masked_depth.png" % (
                        self.save_dir, self.iter),
                    depth_image, self.near_distance,
                    self.far_distance)

        # Apply normal limiting
        if self.rgbd_normal_limit > 0.:
            depth_image *= normalThresh

        # And finally apply rectification to RGB frame
        if self.rgbd_rectification_baseline != 0.:
            # Convert depth image to point cloud, with +z being
            # camera "forward"
            K = self.camera.depth_camera_info().intrinsic_matrix()
            K_rgb = self.camera.color_camera_info().intrinsic_matrix()
            Kinv = np.linalg.inv(K)
            U, V = np.meshgrid(np.arange(h), np.arange(w))
            points_in_camera_frame = np.vstack([
                U.T.flatten(),
                V.T.flatten(),
                np.ones(w*h)])
            points_in_camera_frame = Kinv.dot(points_in_camera_frame) * \
                depth_image.flatten()
            # Shift them over into the rgb camera frame
            # points_in_camera_frame[0, :] += self.rgbd_rectification_baseline
            # Reproject back into the the image (using the RGB
            # projection matrix. This is wrong (and very bad) if the
            # resolution of the RGB and Depth are different.
            points_in_camera_frame[1, :] += self.rgbd_rectification_baseline
            points_in_camera_frame[0, :] /= points_in_camera_frame[2, :]
            points_in_camera_frame[1, :] /= points_in_camera_frame[2, :]
            points_in_camera_frame[2, :] /= points_in_camera_frame[2, :]
            depth_image_out = np.full(depth_image.shape, np.inf)
            reprojected_uv = K_rgb.dot(points_in_camera_frame)
            for u in range(h):
                for v in range(w):
                    if not np.isfinite(reprojected_uv[2, u*w+v]):
                        continue
                    proj_vu = np.round(reprojected_uv[:, u*w+v]).astype(int)
                    if (proj_vu[0] >= 0 and proj_vu[0] < h and
                            proj_vu[1] >= 0 and proj_vu[1] < w):
                        depth_image_out[proj_vu[0], proj_vu[1]] = (
                            min(depth_image_out[proj_vu[0], proj_vu[1]],
                                depth_image[u, v]))
            # Resaturate infs to 0
            depth_image_out[np.isinf(depth_image_out)] = 0.

        else:
            depth_image_out = depth_image

        save_depth_colormap(
            "%s/%05d_masked_depth.png" % (
                self.save_dir, self.iter),
            depth_image_out, self.near_distance,
            self.far_distance)

        # Where it's infinite, set to 0
        depth_image_out = np.where(
            good_mask, depth_image_out,
            np.zeros(depth_image.shape))

        y_data.get_mutable_value().mutable_data[:, :, 0] = \
            depth_image_out[:, :]
        print "Elapsed in render (model): %f seconds" % \
            (time.time() - start_time)
        self.iter += 1


class DepthImageCNNCorruptionBlock(DepthImageCorruptionBlock):
    def __init__(self, camera, save_dir, single_scene_mode=False):
        DepthImageCorruptionBlock.__init__(self, camera, save_dir)
        self.set_name('depth image corruption, cnn')

        if single_scene_mode:
            self.model = network.load_trained_model(
                weights_path="DepthSim/python/models/"
                "2017-06-16-30_depth_batch4.hdf5")
        else:
            self.model = network.load_trained_model(
                weights_path="DepthSim/python/models/net_depth_seg_v1.hdf5")

        self.near_distance = 0.2
        self.far_distance = 3.5
        self.dropout_threshold = 0.15
        self.iter = 0

    def _DoCalcAbstractOutput(self, context, y_data):
        start_time = time.time()

        u_data = self.EvalAbstractInput(context, 1).get_value()
        h, w, _ = u_data.data.shape
        rgb_image = np.empty((h, w), dtype=np.float64)
        rgb_image[:, :] = u_data.data[:, :, 0]

        if self.save_dir is not None:
            save_image_uint8(
                "%s/%05d_rgb.png" % (self.save_dir, self.iter), rgb_image)

        u_data = self.EvalAbstractInput(context, 0).get_value()
        h, w, _ = u_data.data.shape
        depth_image = np.empty((h, w), dtype=np.float64)
        depth_image[:, :] = u_data.data[:, :, 0]
        good_mask = np.isfinite(depth_image)
        depth_image = np.clip(depth_image, self.near_distance,
                              self.far_distance)

        depth_image_normalized = depth_image / self.far_distance
        depth_image_resized = cv2.resize(
            depth_image_normalized, (640, 480),
            interpolation=cv2.INTER_NEAREST)


        stack = np.empty((1, 480, 640, 1))
        stack[0, :, :, 0] = depth_image_resized[:, :]
        predicted_prob_map = self.model.predict_on_batch(stack)

        if self.save_dir is not None:
            save_depth_colormap(
                "%s/%05d_input_depth.png" % (self.save_dir, self.iter),
                depth_image_resized, self.near_distance/self.far_distance, 1.0)
            save_image_colormap(
                "%s/%05d_mask.png" % (self.save_dir, self.iter),
                predicted_prob_map[0, :, :, 0])

        depth_image_resized[predicted_prob_map[0, :, :, 0] >
                            self.dropout_threshold] = 0.
        # Reason about low-probability dropout in superpixel-sized regions,
        # since dropouts often occur on block level
        #blocks_scale_factor = 8
        #prob_map_maxpool = sp.ndimage.filters.maximum_filter(
        #    predicted_prob_map[0, :, :, 0], size=blocks_scale_factor,
        #    mode="nearest")
        #dropouts_small = cv2.resize(
        #    prob_map_maxpool,
        #    (640 / blocks_scale_factor, 480 / blocks_scale_factor),
        #    interpolation=cv2.INTER_NEAREST)
        #noise_matrix = np.random.random(dropouts_small.shape)
        #mask = cv2.resize(1.0*(dropouts_small > noise_matrix), (640, 480),
        #                  interpolation=cv2.INTER_LINEAR)
        #depth_image_resized[mask > 0.5] = 0.

        #save_image_colormap("%s/%05d_noise_matrix.png" % (self.save_dir, self.iter),
        #                    mask)
        #network.apply_mask(predicted_prob_map, depth_image_resized,
        #                   self.dropout_threshold)
        #depth_image_resized = np.where(
        #    predicted_prob_map[0, :, :, 0] <= self.dropout_threshold,
        #    depth_image_resized,
        #    np.zeros(depth_image_resized.shape))
        depth_image = self.far_distance * depth_image_resized

        if self.save_dir is not None:
            save_depth_colormap(
                "%s/%05d_masked_depth.png" % (self.save_dir, self.iter),
                depth_image, self.near_distance, self.far_distance)

        # Where it's infinite, set to 0
        depth_image = np.where(
            good_mask, depth_image,
            np.zeros(depth_image.shape))
        y_data.get_mutable_value().mutable_data[:, :, 0] = \
            depth_image[:, :]
        print "Elapsed in render (cnn): %f seconds" % \
            (time.time() - start_time)
        self.iter += 1


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
                        default=4.0)
    parser.add_argument("--test",
                        action="store_true",
                        help="Help out CI by launching a meshcat server for "
                             "the duration of the test.")
    parser.add_argument("--corruption_method",
                        type=str,
                        default="cnn",
                        help="[cnn, cnn_single_scene, model, none]")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for rng, "
                             "including scene generation.")
    parser.add_argument("--save_dir",
                        type=str,
                        default=None,
                        help="Directory to save depth diagnostic images."
                             " If not specified, does not save images.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

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
    q0 = setup_tabletop(rbt)

    # Set up a visualizer for the robot
    pbrv = MeshcatRigidBodyVisualizer(rbt, draw_timestep=0.01)
    # (wait while the visualizer warms up and loads in the models)
    time.sleep(2.0)

    # Make our RBT into a plant for simulation
    rbplant = RigidBodyPlant(rbt)
    rbplant.set_name("Rigid Body Plant")
    allmaterials = CompliantMaterial()
    allmaterials.set_youngs_modulus(1E8)  # default 1E9
    allmaterials.set_dissipation(1.0)     # default 0.32
    allmaterials.set_friction(0.9)        # default 0.9.
    rbplant.set_default_compliant_material(allmaterials)

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
                   z_near=0.2, z_far=3.5, fov_y=np.pi / 4,
                   width=640, height=480,
                   show_window=False))
    builder.Connect(rbplant_sys.state_output_port(),
                    camera.get_input_port(0))

    if args.corruption_method == "model":
        depth_corruptor = builder.AddSystem(
            DepthImageHeuristicCorruptionBlock(camera, args.save_dir))
        builder.Connect(camera.color_image_output_port(),
                        depth_corruptor.color_image_input_port)
        builder.Connect(camera.depth_image_output_port(),
                        depth_corruptor.depth_image_input_port)
        final_depth_output_port = depth_corruptor.depth_image_output_port
    elif (args.corruption_method == "cnn" or
          args.corruption_method == "cnn_single_scene"):
        depth_corruptor = builder.AddSystem(
            DepthImageCNNCorruptionBlock(
                camera, args.save_dir,
                single_scene_mode=(
                    args.corruption_method == "cnn_single_scene")))
        builder.Connect(camera.color_image_output_port(),
                        depth_corruptor.color_image_input_port)
        builder.Connect(camera.depth_image_output_port(),
                        depth_corruptor.depth_image_input_port)
        final_depth_output_port = depth_corruptor.depth_image_output_port
    elif args.corruption_method == "none":
        final_depth_output_port = camera.depth_image_output_port()
    else:
        print "Got invalid corruption method %s." % args.corruption_method
        sys.exit(-1)

    camera_meshcat_visualizer = builder.AddSystem(
        RgbdCameraMeshcatVisualizer(camera, rbt))
    builder.Connect(final_depth_output_port,
                    camera_meshcat_visualizer.camera_input_port)
    builder.Connect(rbplant_sys.state_output_port(),
                    camera_meshcat_visualizer.state_input_port)

    # Done!
    diagram = builder.Build()

    # Create a simulator for it.
    simulator = Simulator(diagram)

    # The simulator simulates forward from a given Context,
    # so we adjust the simulator's initial Context to set up
    # the initial state.
    state = simulator.get_mutable_context().\
        get_mutable_continuous_state_vector()
    x0 = np.zeros(rbplant_sys.get_num_states())
    x0[0:q0.shape[0]] = q0
    state.SetFromVector(x0)

    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    # Simulator time steps will be very small, so don't
    # force the rest of the system to update every single time.
    simulator.set_publish_every_time_step(False)

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
