import gymnasium as gym
import numpy as np
from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    RollPitchYaw,
    Solve,
)


class ManipulatorEEFWrapper(gym.Wrapper):
    def __init__(self, env, theta_bound=1e-4):
        super().__init__(env)

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
        self.action_space = gym.spaces.Dict(
            {
                "eef_pos": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float),
                "eef_ori": gym.spaces.Box(
                    low=-np.pi, high=np.pi, shape=(3,), dtype=float
                ),
            }
        )
        self.theta_bound = theta_bound

    def eefpose2config(self, eefpose):
        eef_transform = RigidTransform(
            RotationMatrix(RollPitchYaw(eefpose[3:])), eefpose[:3]
        )
        self.end_effector_frame = self.unwrapped.end_effector_body.body_frame()
        ik = InverseKinematics(self.unwrapped.plant, self.unwrapped.plant_context)
        ik.AddPositionConstraint(
            frameB=self.end_effector_frame,
            p_BQ=[0, 0, 0],
            frameA=self.unwrapped.plant.world_frame(),
            p_AQ_lower=eef_transform.translation() - 1e-4,
            p_AQ_upper=eef_transform.translation() + 1e-4,
        )
        ik.AddOrientationConstraint(
            frameAbar=self.end_effector_frame,
            R_AbarA=eef_transform.rotation(),
            frameBbar=self.unwrapped.plant.world_frame(),
            R_BbarB=RotationMatrix(),
            theta_bound=self.theta_bound,
        )
        prog = ik.prog()
        prog.SetInitialGuess(
            ik.q(), self.env.plant.GetPositions(self.env.plant_context)
        )
        result = Solve(prog)
        if not result.is_success():
            raise RuntimeError("Inverse kinematics failed")
        # TODO: Handle IK when env_objects are present
        return result.GetSolution(ik.q())[: self.unwrapped.num_dof]

    def step(self, action):
        eef_pos = action["eef_pos"]
        eef_ori = action["eef_ori"]
        eefpose = np.concatenate((eef_pos, eef_ori))

        q_desired = self.eefpose2config(eefpose)
        obs_in, reward, terminated, truncated, info_in = self.env.step(q_desired)
        obs = {
            "eef_pos": info_in["eef_pos"],
            "eef_quat": info_in["eef_quat"],
            "eef_pos_vel": info_in["eef_pos_vel"],
            "eef_rot_vel": info_in["eef_rot_vel"],
        }
        info = {
            "robot_joint_pos": obs_in["robot_joint_pos"],
            "robot_joint_vel": obs_in["robot_joint_vel"],
            "timestamp": info_in["timestamp"],
        }

        return obs, reward, terminated, truncated, info
