from dataclasses import dataclass
import numpy.typing as npt
from drake import lcmt_viewer_geometry_data

@dataclass
class Geom:
    """
    A data class representing geometric data for visualization in a USD stage.

    Attributes:
        name (str): The name identifier for the geometry.
        position (npt.ArrayLike): The position of the geometry in 3D space.
        quaternion (npt.ArrayLike): The orientation of the geometry as a quaternion.
        color (npt.ArrayLike): The color of the geometry.
        string_data (str): Path to the USDA file generated from a GLTF file.
        stage_path (str): The USD stage path where the geometry will be placed.
        should_add (bool): Flag indicating whether to add the geometry to the stage.
    """

    name: str
    position: npt.ArrayLike
    quaternion: npt.ArrayLike
    color: npt.ArrayLike

    @staticmethod
    def from_geometry_data(
        msg: lcmt_viewer_geometry_data,
        root: str = "/World/",
        name: str = "",
    ) -> "Geom":
        """
        Create a Geom instance from geometry data received from Drake's visualization.

        Args:
            msg (lcmt_viewer_geometry_data): The geometry data message from Drake.
            root (str, optional): The root path in the USD stage. Defaults to "/World/".
            name (str, optional): The name identifier for the geometry. Defaults to "".

        Returns:
            Geom: An instance of the Geom class populated with the provided data.
        """

        return Geom(
            name=name,
            position=msg.position,
            quaternion=msg.quaternion,
            color=msg.color,
        )
