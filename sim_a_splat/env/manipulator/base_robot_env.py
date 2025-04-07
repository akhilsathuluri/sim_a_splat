# %%

from abc import ABC, abstractmethod
import numpy as np
from drake import (
    lcmt_viewer_draw,
    lcmt_viewer_load_robot,
    lcmt_viewer_link_data,
    lcmt_viewer_geometry_data,
)


# %%


class BaseRobotEnv(ABC):
    def __init__(
        self,
        env_objects: bool,
        visualise_flag: bool,
        eef_link_name: str,
        package_path: str,
        package_name: str,
        urdf_name: str,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def reset(self, reset_to_state=None):
        pass

    @abstractmethod
    def set_joint_vector_in_drake(self, pos: np.ndarray):
        pass

    @abstractmethod
    def set_visualize_robot_flag(self, visualize_robot: bool):
        pass

    @abstractmethod
    def step(self, action: np.ndarray, no_obs: bool):
        pass

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def _get_action(self):
        pass

    @abstractmethod
    def _get_info(self):
        pass

    @abstractmethod
    def _compute_reward(self, info):
        pass

    @abstractmethod
    def _is_done(self, info, reward) -> bool:
        pass

    @abstractmethod
    def _set_to_state(self, state):
        pass

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def close(self):
        self.reset()

    @abstractmethod
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    @abstractmethod
    def visualize_robot(self):
        pass

    @abstractmethod
    def get_simulation_time(self):
        pass

    @abstractmethod
    def get_simulation_frequency(self):
        pass

    @abstractmethod
    def close_visualization(self):
        pass

    @abstractmethod
    def render(self):
        pass
