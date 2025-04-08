# %%

from pathlib import Path
import logging

import numpy as np
from pydrake.all import (
    RigidTransform,
    RollPitchYaw,
    TriangleSurfaceMesh,
    RotationMatrix,
    DrakeLcm,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    ContactModel,
    CollisionFilterDeclaration,
    GeometrySet,
    SurfaceTriangle,
    Simulator,
    Quaternion,
    StartMeshcat,
    AddDefaultVisualization,
    Rgba,
    Cylinder,
    FixedOffsetFrame,
    SpatialInertia,
    Sphere,
    Box,
)
import open3d as o3d
from pydrake.systems.primitives import Multiplexer, ConstantVectorSource
from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.systems.primitives import PassThrough
from drake import (
    lcmt_viewer_draw,
    lcmt_viewer_load_robot,
    lcmt_viewer_link_data,
    lcmt_viewer_geometry_data,
)

from sim_a_splat.env.xarm.xarm_sim_utils import (
    PoseToConfig,
    add_ground_with_friction,
    add_soft_collisions,
    AddRobotModel,
    add_env_objects,
    # add_cube,
    MakeHardwareStation,
)


# %%


class XarmGripperSimEnv:
    def __init__(
        self,
        env_objects=True,
        visualise_flag=True,
        eef_link_name=None,
        package_path=None,
        package_name=None,
        urdf_name=None,
    ):
        self.active_meshcat = False
        self.time_step = 1e-2
        self.visualize_robot_flag = False
        self.lcm = DrakeLcm()
        self.env_objects_flag = env_objects
        self.set_visualize_robot_flag(visualise_flag)
        self.seed()
        self.eef_link_name = eef_link_name
        self.package_path = package_path
        self.package_name = package_name
        self.urdf_name = urdf_name

    def load_model(self):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self.time_step
        )
        self.scene_graph = scene_graph
        if self.env_objects_flag:
            x_mid = 0.475
            y_mid = 0.0
            cube_size = 0.050
            small_offset = 0.002
            # red_cube = add_cube(
            #     plant,
            #     "red_cube",
            #     np.array([1, 0, 0, 1]),
            #     np.array([x_mid, y_mid, 5 * cube_size / 2 + 3 * small_offset]),
            #     cube_size=cube_size,
            # )
            # blue_cube = add_cube(
            #     plant,
            #     "blue_cube",
            #     np.array([0, 0, 1, 1]),
            #     np.array([x_mid, y_mid, 3 * cube_size / 2 + 2 * small_offset]),
            #     cube_size=cube_size,
            # )
            # green_cube = add_cube(
            #     plant,
            #     "green_cube",
            #     np.array([0, 1, 0, 1]),
            #     np.array([x_mid, y_mid, cube_size / 2 + small_offset]),
            #     cube_size=cube_size,
            # )
            green_cube = add_env_objects(
                plant=plant,
                scene_graph=scene_graph,
                object_path="assets/bricks/foam_brick.sdf",
                init_pose=RigidTransform(
                    RotationMatrix(),
                    np.array([x_mid, y_mid, cube_size / 2 + small_offset]),
                ),
                prefix="green_cube",
            )
            # blue_cube = add_env_objects(
            #     plant=plant,
            #     scene_graph=scene_graph,
            #     object_path="assets/bricks/foam_brick.sdf",
            #     init_pose=RigidTransform(
            #         RotationMatrix(),
            #         np.array([x_mid, y_mid, 3 * cube_size / 2 + 2 * small_offset]),
            #     ),
            #     prefix="blue_cube",
            # )

            pass
        self.robot_model_instance, self.uid = AddRobotModel(
            plant=plant,
            scene_graph=scene_graph,
            package_path=self.package_path,
            package_name=self.package_name,
            urdf_name=self.urdf_name,
            weld_frame_transform=RigidTransform(),
        )

        eef_base_link = plant.GetBodyByName(
            "xarm_gripper_base_link", self.robot_model_instance
        )

        self.eef_link_name = "gripper_center"
        self.end_effector_frame = plant.AddFrame(
            FixedOffsetFrame(
                name="gripper_center",
                P=eef_base_link.body_frame(),
                X_PF=RigidTransform(RotationMatrix(), np.array([0.0, 0.0, 0.15])),
                model_instance=self.robot_model_instance,
            ),
        )
        # plant.set_contact_model(ContactModel.kHydroelasticsOnly)
        plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
        add_ground_with_friction(plant)
        plant.set_penetration_allowance(1e-4)
        # collision_filter_manager = scene_graph.collision_filter_manager()
        # gripper_finger_names = ["left_finger", "right_finger"]
        add_soft_collisions(
            plant,
            "right_finger",
            Box(0.03, 0.010, 0.02),
            # Sphere(0.015),
            offset=RigidTransform(np.array([0.0, 0.02, 0.05])),
        )
        add_soft_collisions(
            plant,
            "left_finger",
            Box(0.03, 0.010, 0.02),
            # Sphere(0.015),
            offset=RigidTransform(np.array([0.0, -0.02, 0.05])),
        )
        # for ii in gripper_finger_names:
        #     collision_filter_manager.Apply(
        #         CollisionFilterDeclaration().ExcludeBetween(
        #             GeometrySet(
        #                 plant.GetCollisionGeometriesForBody(
        #                     plant.GetBodyByName(
        #                         ii,
        #                         self.robot_model_instance,
        #                     )
        #                 )
        #             ),
        #             GeometrySet(
        #                 plant.GetCollisionGeometriesForBody(plant.world_body())
        #             ),
        #         )
        #     )
        plant.Finalize()
        self.nq = plant.num_positions(self.robot_model_instance)
        station = builder.AddSystem(
            MakeHardwareStation(
                self.time_step,
                package_path=self.package_path,
                package_name=self.package_name,
                urdf_name=self.urdf_name,
                weld_frame_transform=RigidTransform(),
                uid=self.uid,
            )
        )
        pose2config = builder.AddSystem(PoseToConfig(plant, self.end_effector_frame))
        builder.Connect(
            plant.get_state_output_port(self.robot_model_instance),
            station.GetInputPort("robot_estimated_state"),
        )
        builder.ExportOutput(plant.get_state_output_port(), "system_state_output_port")
        builder.Connect(
            station.GetOutputPort("robot_torque_commanded"),
            plant.get_actuation_input_port(self.robot_model_instance),
        )

        gripper_system = builder.AddSystem(PassThrough(2))
        builder.ExportInput(gripper_system.get_input_port(), "gripper_command")
        joint_pos_size = pose2config.get_output_port().size()
        mux = builder.AddSystem(Multiplexer([joint_pos_size, 2]))

        # Connect the systems
        builder.Connect(pose2config.get_output_port(), mux.get_input_port(0))
        builder.Connect(gripper_system.get_output_port(), mux.get_input_port(1))

        builder.Connect(
            mux.get_output_port(), station.GetInputPort("robot_state_desired")
        )
        builder.ExportInput(pose2config.GetInputPort("pose"), "desired_pose")
        builder.ExportOutput(
            pose2config.GetOutputPort("config"), "desired_joint_position"
        )
        if self.visualize_robot_flag:
            AddDefaultVisualization(builder=builder, meshcat=self.meshcat)

        self.plant = plant
        self.diagram = builder.Build()
        self.simulator = Simulator(self.diagram)
        self.pose_input_port = self.simulator.get_system().GetInputPort("desired_pose")
        self.gripper_input_port = self.simulator.get_system().GetInputPort(
            "gripper_command"
        )
        self.state_output_port = self.simulator.get_system().GetOutputPort(
            "system_state_output_port"
        )
        self.desired_joint_position = self.simulator.get_system().GetOutputPort(
            "desired_joint_position"
        )

    def reset(self, reset_to_state=None):
        self.diagram_default_context = self.diagram.CreateDefaultContext()
        self.simulator.reset_context(self.diagram_default_context)
        self.simulator_context = self.simulator.get_mutable_context()
        self.diagram_context = self.diagram.GetMyContextFromRoot(self.simulator_context)
        self.plant_context = self.plant.GetMyMutableContextFromRoot(
            self.simulator_context
        )
        if reset_to_state is None:
            reset_to_state = [
                self.np_random.uniform(
                    low=np.array([0.25, -0.3, 0.2]), high=np.array([0.65, 0.3, 0.2])
                ),
                self.np_random.uniform(
                    low=np.array([0.4, -0.183, 0.2, -np.pi]),
                    high=np.array([0.55, 0.183, 0.2, np.pi]),
                ),
                np.array([0.475, 0.0, 0.2, 0.78539816]),
            ]
        self.pose_input_port.FixValue(
            self.diagram_context,
            RigidTransform(RollPitchYaw(3.14, 0, 0), reset_to_state[0]),
        )
        self.gripper_input_port.FixValue(
            self.diagram_context, self.np_random.uniform([-0.045, -0.045], [0, 0])
        )
        if self.env_objects_flag:
            pass
        jpos = self.desired_joint_position.Eval(self.diagram_context)
        eefpos = self.gripper_input_port.Eval(self.diagram_context)
        robotpos = np.concatenate((jpos, eefpos))
        self.plant.SetPositions(
            self.plant_context,
            self.robot_model_instance,
            robotpos,
        )
        self.plant.SetVelocities(
            self.plant_context,
            self.robot_model_instance,
            np.zeros(len(robotpos)),
        )
        # reset_to_state[2][2] = 0
        # self.goal_pose_transform = RigidTransform(
        #     RotationMatrix(RollPitchYaw(0, 0, -reset_to_state[2][3])),
        #     reset_to_state[2][:3],
        # )
        # self.publish_tblock_marker(self.goal_pose_transform, color=Rgba(0, 1, 0, 0.2))
        self.simulator.Initialize()

    def set_joint_vector_in_drake(self, pos):
        self.qpos = pos

    def set_visualize_robot_flag(self, visualize_robot):
        self.visualize_robot_flag = visualize_robot
        if self.visualize_robot_flag and not self.active_meshcat:
            logging.info(
                "Visualization for drake via meshcat can be accessed via the printed URL."
            )
            self.meshcat = StartMeshcat()
            self.active_meshcat = True

    def publish_robot_end_location(
        self, end_location=np.array([0.25, 0.3, 0.2]), color=Rgba(0, 0, 1, 0.2)
    ):
        if self.visualize_robot_flag and self.active_meshcat:
            radius, length = 0.013, 0.05
            self.meshcat.SetObject(
                "eef_goal", Cylinder(radius=radius, length=length), color
            )
            self.meshcat.SetTransform("eef_goal", RigidTransform(end_location))

    def step(self, action, no_obs=False):
        self.pose_input_port.FixValue(
            self.diagram_context, RigidTransform(RollPitchYaw(3.14, 0, 0), action[:3])
        )
        self.gripper_input_port.FixValue(self.diagram_context, action[3:])
        try:
            self.simulator.AdvanceTo(self.simulator_context.get_time() + self.time_step)
        except RuntimeError as e:
            logging.error(f"Drake simulator failed to advance: {e}")
        observation = self._get_obs()
        info = self._get_info()
        reward = 0.0
        done = False
        return observation, reward, done, info

    def _get_obs(self):
        eef_pose = self.plant.CalcRelativeTransform(
            self.plant_context,
            self.plant.world_frame(),
            self.end_effector_frame,
        )

        eef_pos = eef_pose.translation()
        eef_quat = eef_pose.rotation().ToQuaternion().wxyz()
        eef_vel = self.end_effector_frame.CalcRelativeSpatialVelocityInWorld(
            self.plant_context, self.plant.world_frame()
        )
        return (
            eef_pos,
            eef_quat,
            eef_vel.translational(),
            eef_vel.rotational(),
        )

    def _get_action(self):
        desired_eef_pose = self.pose_input_port.Eval(self.diagram_context)
        desired_gripper_state = self.gripper_input_port.Eval(self.diagram_context)
        return np.concatenate((desired_eef_pose.translation(), desired_gripper_state))

    def _get_info(self):
        robot_state = self.plant.get_state_output_port(self.robot_model_instance).Eval(
            self.plant_context
        )
        robot_pos = robot_state[: self.nq]
        robot_vel = robot_state[self.nq :]
        eef_pose = self.plant.CalcRelativeTransform(
            self.plant_context,
            self.plant.world_frame(),
            self.end_effector_frame,
        )

        eef_pos = eef_pose.translation()
        eef_quat = eef_pose.rotation().ToQuaternion().wxyz()
        eef_vel = self.end_effector_frame.CalcRelativeSpatialVelocityInWorld(
            self.plant_context, self.plant.world_frame()
        )

        info = {
            "robot_pos": robot_pos,
            "robot_vel": robot_vel,
            "block_pose": np.array([1, 0, 0, 0, 0, 0, 0]),
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "eef_vel": eef_vel.translational(),
            "eef_ang_vel": eef_vel.rotational(),
            "timestamp": self.simulator_context.get_time(),
        }
        return info

    def _compute_reward(self, info):
        return 0.0
        pass

    def _is_done(self, info, reward):
        if abs(reward) < 0.02:
            return True
        return False

    def _set_to_state(self, state):
        self.plant.SetPositions(
            self.plant_context,
            self.robot_model_instance,
            state["robot_pos"],
        )
        self.plant.SetVelocities(
            self.plant_context,
            self.robot_model_instance,
            np.zeros(len(state["robot_pos"])),
        )
        if self.env_objects_flag:
            pass
        self.simulator_context.SetTime(state["timestamp"])

    def _generate_loader_msg(self):
        loader_msg = lcmt_viewer_load_robot()
        mi = self.scene_graph.model_inspector()
        frame_ids = mi.GetAllFrameIds()
        for ii in range(len(frame_ids)):
            if mi.GetFrameGroup(frame_ids[ii]) >= 0:
                link_msg = lcmt_viewer_link_data()
                link_msg.robot_num = mi.GetFrameGroup(frame_ids[ii])
                link_msg.name = "plant::" + mi.GetName(frame_ids[ii])
                geometry_ids = mi.GetGeometries(frame_ids[ii])
                for gi in geometry_ids:
                    geom_msg = lcmt_viewer_geometry_data()
                    pose = mi.GetPoseInFrame(gi)
                    geom_msg.position = pose.translation()
                    geom_msg.quaternion = pose.rotation().ToQuaternion().wxyz()
                    link_msg.num_geom += 1
                loader_msg.link.append(link_msg)
                loader_msg.num_links += 1
        return loader_msg

    def _generate_draw_msg(self):
        draw_msg = lcmt_viewer_draw()
        mi = self.scene_graph.model_inspector()
        frame_ids = mi.GetAllFrameIds()
        for ii in range(len(frame_ids)):
            if mi.GetFrameGroup(frame_ids[ii]) > 0:
                body = self.plant.GetBodyFromFrameId(frame_ids[ii])
                draw_msg.robot_num.append(int(body.model_instance()))
                draw_msg.link_name.append("plant::" + body.name())
                pose = self.plant.EvalBodyPoseInWorld(self.plant_context, body)
                draw_msg.position.append(pose.translation())
                draw_msg.quaternion.append(pose.rotation().ToQuaternion().wxyz())
                draw_msg.num_links += 1
        return draw_msg

    def close(self):
        self.reset()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def visualize_robot(self):
        pass

    def get_simulation_time(self):
        return self.simulator_context.get_time()

    def get_simulation_frequency(self):
        return self.time_step

    def close_visualization(self):
        self.meshcat.Delete()
        logging.info("Close the browser tab to stop visualization.")
        self.reset()

    def render(self):
        self.diagram.ForcedPublish(self.simulator_context)
