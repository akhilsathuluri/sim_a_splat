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
    SurfaceTriangle,
    Simulator,
    Quaternion,
    StartMeshcat,
    AddDefaultVisualization,
    Rgba,
    Cylinder,
)
import open3d as o3d
from drake import (
    lcmt_viewer_draw,
    lcmt_viewer_load_robot,
    lcmt_viewer_link_data,
    lcmt_viewer_geometry_data,
)

from sim_a_splat.env.manipulator.manipulator_sim_utils import (
    PoseToConfig,
    AddRobotModel,
    add_env_objects,
    MakeHardwareStation,
    configure_contacts,
)

from typing import Optional
import gymnasium as gym


# %%


class ManipulatorSimEnv(gym.Env):
    def __init__(
        self,
        env_objects=True,
        visualise_flag=True,
        eef_link_name=None,
        package_path=None,
        package_name=None,
        urdf_name=None,
        num_dof=None,
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
        self.num_dof = num_dof

        self.observation_space = gym.spaces.Dict(
            {
                "eef_pos": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "eef_quat": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,),
                    dtype=np.float32,
                ),
                "eef_pos_vel": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "eef_rot_vel": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )

    def load_model(self):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self.time_step
        )
        self.scene_graph = scene_graph
        if self.env_objects_flag:
            tblock = add_env_objects(plant, scene_graph)
        self.robot_model_instance, self.uid = AddRobotModel(
            plant=plant,
            scene_graph=scene_graph,
            package_path=self.package_path,
            package_name=self.package_name,
            urdf_name=self.urdf_name,
            weld_frame_transform=RigidTransform(),
        )
        configure_contacts(
            plant,
            eef_link_name=self.eef_link_name,
            scene_graph=scene_graph,
            robot_model_instance=self.robot_model_instance,
        )
        plant.Finalize()
        self.nq = plant.num_positions(self.robot_model_instance)
        self.end_effector_body = plant.GetBodyByName(self.eef_link_name)
        self.end_effector_frame = self.end_effector_body.body_frame()
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
        pose2config = builder.AddSystem(
            PoseToConfig(plant, self.end_effector_frame, self.nq)
        )
        builder.Connect(
            plant.get_state_output_port(self.robot_model_instance),
            station.GetInputPort("robot_estimated_state"),
        )
        builder.ExportOutput(plant.get_state_output_port(), "system_state_output_port")
        builder.Connect(
            station.GetOutputPort("robot_torque_commanded"),
            plant.get_actuation_input_port(self.robot_model_instance),
        )
        builder.Connect(
            pose2config.get_output_port(), station.GetInputPort("robot_state_desired")
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
        self.state_output_port = self.simulator.get_system().GetOutputPort(
            "system_state_output_port"
        )
        self.desired_joint_position = self.simulator.get_system().GetOutputPort(
            "desired_joint_position"
        )

    def reset(self, seed: Optional[int] = None, reset_to_state=None):
        super().reset(seed=seed)
        self.diagram_default_context = self.diagram.CreateDefaultContext()
        self.simulator.reset_context(self.diagram_default_context)
        self.simulator_context = self.simulator.get_mutable_context()
        self.diagram_context = self.diagram.GetMyContextFromRoot(self.simulator_context)
        self.plant_context = self.plant.GetMyMutableContextFromRoot(
            self.simulator_context
        )
        if reset_to_state is None:
            reset_to_state = {
                "eef_pos": self.np_random.uniform(
                    low=np.array([0.25, -0.3, 0.2]), high=np.array([0.65, 0.3, 0.2])
                ),
                "block_pos": self.np_random.uniform(
                    low=np.array([0.4, -0.183, 0.2, -np.pi]),
                    high=np.array([0.55, 0.183, 0.2, np.pi]),
                ),
                "goal_pos": np.array([0.475, 0.0, 0.2, 0.78539816]),
            }
        self.pose_input_port.FixValue(
            self.diagram_context,
            RigidTransform(RollPitchYaw(3.14, 0, 0), reset_to_state["eef_pos"][:3]),
        )

        if self.env_objects_flag:
            reset_to_state["block_pos"][2] = 0
            block_pose = np.hstack(
                (
                    RotationMatrix(RollPitchYaw(0, 0, -reset_to_state["block_pos"][3]))
                    .ToQuaternion()
                    .wxyz(),
                    reset_to_state["block_pos"][:3],
                )
            )
            self.plant.SetPositions(
                self.plant_context,
                self.plant.GetModelInstanceByName("tblock_paper"),
                block_pose,
            )
            self.plant.SetVelocities(
                self.plant_context,
                self.plant.GetModelInstanceByName("tblock_paper"),
                np.zeros(6),
            )
            reset_to_state["goal_pos"][2] = 0
            self.goal_pose_transform = RigidTransform(
                RotationMatrix(RollPitchYaw(0, 0, -reset_to_state["goal_pos"][3])),
                reset_to_state["goal_pos"][:3],
            )
            self.publish_tblock_marker(
                self.goal_pose_transform, color=Rgba(0, 1, 0, 0.2)
            )

        jpos = self.desired_joint_position.Eval(self.diagram_context)
        self.plant.SetPositions(
            self.plant_context,
            self.robot_model_instance,
            jpos,
        )
        self.plant.SetVelocities(
            self.plant_context,
            self.robot_model_instance,
            np.zeros(len(jpos)),
        )
        self.simulator.Initialize()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

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

    def publish_tblock_marker(self, block_pose_transform, color=Rgba(1, 0, 0, 1)):
        if self.visualize_robot_flag and self.active_meshcat:
            mesh_path = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "assets/tblock_paper/tblock_paper.obj"
            )
            tblock_mesh = o3d.io.read_triangle_mesh(Path(mesh_path).resolve().__str__())
            triangles = np.asarray(tblock_mesh.triangles)
            tblock_mesh_drake = TriangleSurfaceMesh(
                triangles=[
                    SurfaceTriangle(triangle[0], triangle[1], triangle[2])
                    for triangle in triangles
                ],
                vertices=np.asarray(tblock_mesh.vertices),
            )
            self.meshcat.SetObject("block_marker/mesh", tblock_mesh_drake, color)
            self.meshcat.SetTransform("block_marker/mesh", block_pose_transform)

            msg = lcmt_viewer_draw()
            msg.num_links = 1
            msg.link_name = ["block_marker"]
            msg.robot_num = [3]
            msg.position = [block_pose_transform.translation()]
            msg.quaternion = [block_pose_transform.rotation().ToQuaternion().wxyz()]
            self.lcm.Publish("DRAKE_VIEWER_DRAW", msg.encode())

    def step(self, action, no_obs=False):
        self.pose_input_port.FixValue(
            self.diagram_context, RigidTransform(RollPitchYaw(3.14, 0, 0), action)
        )
        try:
            self.simulator.AdvanceTo(self.simulator_context.get_time() + self.time_step)
        except RuntimeError as e:
            logging.error(f"Drake simulator failed to advance: {e}")
        observation = self._get_obs()
        info = self._get_info()
        reward = self._compute_reward(info)
        terminated = self._is_done(info, reward)
        truncated = False
        if terminated:
            end_location = np.array([0.25, 0.3, 0.2])
            self.publish_robot_end_location(end_location=end_location)
            if type(observation) is tuple:
                eef_goal_dist = np.linalg.norm(observation[0][:2] - end_location[:2])
            else:
                eef_goal_dist = np.linalg.norm(observation["eef_pos"] - end_location)
            if eef_goal_dist > 0.008:
                terminated = False
        else:
            try:
                self.meshcat.Delete("eef_goal")
            except:
                pass

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        eef_pose = self.plant.EvalBodyPoseInWorld(
            self.plant_context, self.end_effector_body
        )
        eef_pos = eef_pose.translation()
        eef_quat = eef_pose.rotation().ToQuaternion().wxyz()
        eef_vel = self.plant.EvalBodySpatialVelocityInWorld(
            self.plant_context, self.end_effector_body
        )
        obs = {
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "eef_pos_vel": eef_vel.translational(),
            "eef_rot_vel": eef_vel.rotational(),
        }
        return obs

    def _get_action(self):
        desired_eef_pose = self.pose_input_port.Eval(self.diagram_context)
        return desired_eef_pose.translation()

    def _get_info(self):
        robot_state = self.plant.get_state_output_port(self.robot_model_instance).Eval(
            self.plant_context
        )
        robot_pos = robot_state[: self.nq]
        robot_vel = robot_state[self.nq :]

        if self.env_objects_flag:
            block_state = self.plant.get_state_output_port(
                self.plant.GetModelInstanceByName("tblock_paper")
            ).Eval(self.plant_context)

            block_pose = block_state[:7]
            block_vel = block_state[7:]
            info = {
                "robot_pos": robot_pos,
                "robot_vel": robot_vel,
                "block_pose": block_pose,
                "block_vel": block_vel,
                "timestamp": self.simulator_context.get_time(),
            }
            return info
        else:
            return {
                "robot_pos": robot_pos,
                "robot_vel": robot_vel,
                "timestamp": self.simulator_context.get_time(),
            }

    def _compute_reward(self, info):
        if self.env_objects_flag:
            goal_pos = self.goal_pose_transform.translation()
            block_pos = info["block_pose"][4:]
            r1 = -np.linalg.norm(goal_pos - block_pos)

            goal_yaw = self.goal_pose_transform.rotation().ToRollPitchYaw().vector()[2]
            quat = info["block_pose"][:4]
            block_yaw = (
                RotationMatrix(Quaternion(quat / np.linalg.norm(quat)))
                .ToRollPitchYaw()
                .vector()
            )[2]
            r2 = -np.abs(goal_yaw - block_yaw)
            return r1 + r2
        else:
            return 0.0

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
            self.plant.SetPositions(
                self.plant_context,
                self.plant.GetModelInstanceByName("tblock_paper"),
                state["block_pose"],
            )
            self.plant.SetVelocities(
                self.plant_context,
                self.plant.GetModelInstanceByName("tblock_paper"),
                np.zeros(6),
            )
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
