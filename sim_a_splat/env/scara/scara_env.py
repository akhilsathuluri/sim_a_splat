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
    FixedOffsetFrame,
    Cylinder,
    Box,
)
import open3d as o3d
from drake import (
    lcmt_viewer_draw,
    lcmt_viewer_load_robot,
    lcmt_viewer_link_data,
    lcmt_viewer_geometry_data,
)

from sim_a_splat.env.manipulator.sim_utils import (
    PoseToConfig,
    add_ground_with_friction,
    add_soft_collisions,
    AddRobotModel,
    add_env_objects,
    MakeHardwareStation,
)

# from sim_a_splat.env.manipulator.base_robot_env import BaseRobotEnv
import gymnasium as gym
from gymnasium import spaces

# %%


class ScaraSimEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
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

        # add gym stuff
        self.observation_space= spaces.Dict({
            "robot_eef_pos": spaces.Box(
                low=np.array([0.075, -0.3, 0.0]),
                high=np.array([0.375, 0.3, 0.0]),
                dtype=np.float64,
            ),
            "camera_1": spaces.Box(
                low=0,
                high=255,
                shape=(3, 240, 320),
                dtype=np.uint8,
            )
        })
        self.action_space = spaces.Box(low=np.array([0.075, -0.3, 0.0]), high=np.array([0.375, 0.3, 0.0]), shape=(3,), dtype=np.float32)


    def load_model(self):
        builder = DiagramBuilder()
        self.plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self.time_step
        )
        self.scene_graph = scene_graph
        if self.env_objects_flag:
            _ = add_env_objects(self.plant, scene_graph)
        self.robot_model_instance, self.uid = AddRobotModel(
            plant=self.plant,
            scene_graph=scene_graph,
            package_path=self.package_path,
            package_name=self.package_name,
            urdf_name=self.urdf_name,
            weld_frame_transform=RigidTransform(
                # np.array([0.434320411787011, -1.31922887815084, 0.644967241561405])
                np.array([0.0, 0.0, 0.0]),
            ),
        )
        # TODO: Modify API to enable loading the same robot multiple times
        # self.eef_link_name = self.eef_link_name + "_" + str(self.uid)
        # self.eef_link_name = self.eef_link_name
        # assumes robot model to be a 6DoF robot arm with fixed base in urdf
        # TODO: Create API to easily make wrappers around anytype of robot and with an inverse dynamics controller
        self.plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
        self.plant.set_penetration_allowance(1e-5)
        # add_ground_with_friction(self.plant)
        box_coll_id = add_soft_collisions(
            plant=self.plant,
            link_name="base_link",
            body=Box(0.6, 0.5, 0.05),
            collision_pose=RigidTransform(
                # RotationMatrix(RollPitchYaw(0.0, np.pi / 2, 0.0)),
                np.array([0.1, 0.0, 0.0125]),
            ),
        )
        if self.eef_link_name == "":
            logging.warning("Set the end effector body here")
            eef_base_link = self.plant.GetBodyByName(
                "link_3", self.robot_model_instance
            )

            self.eef_link_name = "gripper_center"
            self.end_effector_frame = self.plant.AddFrame(
                FixedOffsetFrame(
                    name="gripper_center",
                    P=eef_base_link.body_frame(),
                    X_PF=RigidTransform(RotationMatrix(), np.array([0.0, 0.0, 0.0])),
                    model_instance=self.robot_model_instance,
                ),
            )
            self.plant.RegisterVisualGeometry(
                eef_base_link,
                RigidTransform(),
                Cylinder(radius=0.01, length=0.1),
                "gripper_visual",
                np.array([1.0, 1.0, 1.0, 1.0]),
            )
        else:
            add_soft_collisions(
                self.plant,
                link_name=self.eef_link_name,
                body=Cylinder(radius=0.01, length=0.05),
                collision_pose=RigidTransform(
                    RotationMatrix(), np.array([-0.005, 0.005, 0.0])
                ),
            )
            self.end_effector_body = self.plant.GetBodyByName(self.eef_link_name)
            self.end_effector_frame = self.end_effector_body.body_frame()

        # breakpoint()
        collision_filter_manager = scene_graph.collision_filter_manager()
        collision_filter_manager.Apply(
            CollisionFilterDeclaration().ExcludeBetween(
                GeometrySet(
                    self.plant.GetCollisionGeometriesForBody(
                        self.plant.GetBodyByName(
                            self.eef_link_name,
                            self.robot_model_instance,
                        )
                    )
                ),
                GeometrySet(
                    # self.plant.GetCollisionGeometriesForBody(
                    #     self.plant.GetBodyByName("base_link_collision")
                    # )
                    box_coll_id
                ),
            )
        )
        self.plant.Finalize()
        self.nq = self.plant.num_positions(self.robot_model_instance)
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
            PoseToConfig(self.plant, self.end_effector_frame, relax_ori=True)
        )
        builder.Connect(
            self.plant.get_state_output_port(self.robot_model_instance),
            station.GetInputPort("robot_estimated_state"),
        )
        builder.ExportOutput(
            self.plant.get_state_output_port(), "system_state_output_port"
        )
        builder.Connect(
            station.GetOutputPort("robot_torque_commanded"),
            self.plant.get_actuation_input_port(self.robot_model_instance),
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

        self.diagram = builder.Build()
        self.simulator = Simulator(self.diagram)
        self.pose_input_port = self.simulator.get_system().GetInputPort("desired_pose")
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
                    low=np.array([0.075, -0.3, 0.1]), high=np.array([0.375, 0.3, 0.1])
                ),
                self.np_random.uniform(
                    low=np.array([0.1, -0.183, 0.2, -np.pi]),
                    high=np.array([0.3, 0.183, 0.2, np.pi]),
                ),
                np.array([0.225, 0.0, 0.2, 0.78539816]),
            ]
        self.pose_input_port.FixValue(
            self.diagram_context,
            RigidTransform(RollPitchYaw(0, 0, 0), reset_to_state[0]),
        )
        reset_to_state[1][2] = 0.04
        block_pose = np.hstack(
            (
                RotationMatrix(RollPitchYaw(0, 0, -reset_to_state[1][3]))
                .ToQuaternion()
                .wxyz(),
                reset_to_state[1][:3],
            )
        )
        if self.env_objects_flag:
            self.plant.SetPositions(
                self.plant_context,
                self.plant.GetModelInstanceByName("scaled_tblock"),
                block_pose,
            )
            self.plant.SetVelocities(
                self.plant_context,
                self.plant.GetModelInstanceByName("scaled_tblock"),
                np.zeros(6),
            )
            pass
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
        reset_to_state[2][2] = 0.04
        self.goal_pose_transform = RigidTransform(
            RotationMatrix(RollPitchYaw(0, 0, -reset_to_state[2][3])),
            reset_to_state[2][:3],
        )
        self.publish_tblock_marker(self.goal_pose_transform, color=Rgba(0, 1, 0, 0.2))
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

    def publish_tblock_marker(self, block_pose_transform, color=Rgba(1, 0, 0, 1)):
        if self.visualize_robot_flag and self.active_meshcat:
            mesh_path = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "assets/tblock_paper/scaled_tblock.obj"
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
        self.pose_input_port.FixValue(self.diagram_context, RigidTransform(action))
        try:
            self.simulator.AdvanceTo(self.simulator_context.get_time() + self.time_step)
        except RuntimeError as e:
            logging.error(f"Drake simulator failed to advance: {e}")
        observation = self._get_obs()
        info = self._get_info()
        reward = self._compute_reward(info)
        done = self._is_done(info, reward)
        if done:
            end_location = np.array([0.075, 0.0, 0.1])
            self.publish_robot_end_location(end_location=end_location)
            if type(observation) is tuple:
                eef_goal_dist = np.linalg.norm(observation[0][:2] - end_location[:2])
            else:
                eef_goal_dist = np.linalg.norm(
                    observation["robot_eef_pos"] - end_location[:2]
                )
            if eef_goal_dist > 0.008:
                done = False
        else:
            try:
                self.meshcat.Delete("eef_goal")
            except:
                pass
            
        info['is_success'] = done
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
        return desired_eef_pose.translation()

    def _get_info(self):
        robot_state = self.plant.get_state_output_port(self.robot_model_instance).Eval(
            self.plant_context
        )
        robot_pos = robot_state[: self.nq]
        robot_vel = robot_state[self.nq :]

        block_state = self.plant.get_state_output_port(
            self.plant.GetModelInstanceByName("scaled_tblock")
        ).Eval(self.plant_context)

        block_pose = block_state[:7]
        block_vel = block_state[7:]

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
            "block_pose": block_pose,
            "block_vel": block_vel,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "eef_vel": eef_vel.translational(),
            "eef_ang_vel": eef_vel.rotational(),
            "timestamp": self.simulator_context.get_time(),
        }
        return info

    def _compute_reward(self, info):
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
                self.plant.GetModelInstanceByName("scaled_tblock"),
                state["block_pose"],
            )
            self.plant.SetVelocities(
                self.plant_context,
                self.plant.GetModelInstanceByName("scaled_tblock"),
                np.zeros(6),
            )
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
