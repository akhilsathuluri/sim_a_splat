from dataclasses import dataclass, field
from typing import List, Optional
from drake import lcmt_viewer_link_data
from sim_a_splat.messaging.geom import Geom


@dataclass
class Link:
    """
    A data class representing a robotic link composed of multiple geometries for visualization.

    Attributes:
        name (str): The name identifier for the link.
        robot_num (int): The identifier number for the robot to which this link belongs.
        num_geom (int): The number of geometries associated with this link.
        geoms (List[Geom], optional): A list of Geom instances representing the geometries of the link.
            Initialized to an empty list if not provided.
    """

    name: str
    robot_num: int
    num_geom: int
    geoms: Optional[List[Geom]] = field(default_factory=list)

    def add_geom(self, geom: Geom):
        """
        Add a geometry to the link's geometry list.

        Args:
            geom (Geom): An instance of the Geom class to be added to the link.
        """
        self.geoms.append(geom)

    @staticmethod
    def from_link_data(
        msg: lcmt_viewer_link_data,
        root: str = "/World/",
        name: str = "",
    ) -> "Link":
        """
        Create a Link instance from link data received from Drake's visualization.

        Args:
            msg (lcmt_viewer_link_data): The link data message from Drake.
            root (str, optional): The root path in the USD stage where the link will be placed. Defaults to "/World/".
            name (str, optional): The name identifier for the link. Defaults to "".

        Returns:
            Link: An instance of the Link class populated with the provided data.
        """
        link = Link(name=name, robot_num=msg.robot_num, num_geom=msg.num_geom)

        for geom_idx, geom_data in enumerate(msg.geom):
            geom_name = f"link_{link.name}_geom_idx_{geom_idx}"
            geom = Geom.from_geometry_data(
                msg=geom_data,
                root=root,
                name=geom_name,
            )
            link.add_geom(geom)

        return link
