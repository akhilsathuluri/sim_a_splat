# %%
from abc import abstractmethod

from sim_a_splat.env.manipulator.base_robot_env import BaseRobotEnv


# %%
class BaseSplatEnv(BaseRobotEnv):
    def __init__(
        self,
        visualise_sim_flag: bool,
        eef_link_name: str = None,
        package_path: str = None,
        package_name: str = None,
        urdf_name: str = None,
        # default_cam_pose: tf.SE3 = None,
        splat_dir: str = None,
        match_object_name: str = None,
        splat_config_name: str = None,
        task_mesh_name: str = None,
        **kwargs,
    ):
        super().__init__(
            visualise_flag=self.visualise_sim_flag,
            eef_link_name=eef_link_name,
            package_path=package_path,
            package_name=package_name,
            urdf_name=urdf_name,
        )
        self.eef_link_name = eef_link_name
        super(BaseSplatEnv, self).load_model()
        self.ch = self._setup_splats()
        # comes in as a keyword argument
        # cam_pose_01 = tf.SE3(
        #     wxyz_xyz=np.concatenate(
        #         (
        #             np.array([-0.41946813, 0.89955231, -0.11045113, 0.05150421]),
        #             np.array([-0.15, -0.3, -0.05]),
        #         )
        #     )
        # )
        self.fixed_cam_poses = [kwargs["default_cam_pose"]]
        self.cam_poses = []

    @abstractmethod
    def _setup_splats(self):
        pass

    @abstractmethod
    def _setup_cameras(
        self, ch, additional_cam_poses=[], view_cam_idx=-1, render_size=[[240, 320]] * 2
    ):
        pass

    @abstractmethod
    def reset(self, reset_to_state=None):
        pass

    @abstractmethod
    def set_visual_state(self, state):
        pass

    @abstractmethod
    def get_moving_camera_poses(self, msg):
        pass

    @abstractmethod
    def step(self, action, noobs=False):
        pass

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def render(self, mode="rgb_array"):
        pass

    @abstractmethod
    def close(self):
        pass
