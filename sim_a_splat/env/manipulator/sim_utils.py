from pydrake.all import (
    MultibodyPlant,
    Parser,
    RigidTransform,
    RotationMatrix,
    LeafSystem,
    GeometryInstance,
    AbstractValue,
    Solve,
    InverseKinematics,
    ProximityProperties,
    AddContactMaterial,
    CoulombFriction,
    AddCompliantHydroelasticPropertiesForHalfSpace,
    AddCompliantHydroelasticProperties,
    HalfSpace,
    Cylinder,
    PlanarJoint,
    RobotDiagramBuilder,
    InverseDynamicsController,
    StateInterpolatorWithDiscreteDerivative,
    Frame,
    SpatialInertia,
    UnitInertia,
    Mesh,
)
import numpy as np
from sak.URDFutils import URDFutils
from pathlib import Path
import logging
import tempfile
import xml.etree.ElementTree as ET
import shutil
import meshio
import trimesh


class PoseToConfig(LeafSystem):
    def __init__(self, plant: MultibodyPlant, frame: Frame, relax_ori=False):
        LeafSystem.__init__(self)
        self.plant = plant
        self.frame = frame
        self.relax_ori = relax_ori
        self.plant_context = plant.CreateDefaultContext()
        self.DeclareAbstractInputPort(
            "pose",
            AbstractValue.Make(RigidTransform()),
        )
        self.out_port_len = plant.num_actuators()
        self.DeclareVectorOutputPort(
            "config",
            self.out_port_len,
            self._CalcOutput,
        )

    def _CalcOutput(self, context, output):
        desired_pose = self.get_input_port().Eval(context)
        ik = InverseKinematics(self.plant, self.plant_context)
        ik.AddPositionConstraint(
            frameB=self.frame,
            p_BQ=[0, 0, 0],
            frameA=self.plant.world_frame(),
            p_AQ_lower=desired_pose.translation() - 1e-4,
            p_AQ_upper=desired_pose.translation() + 1e-4,
        )
        if not self.relax_ori:
            ik.AddOrientationConstraint(
                frameAbar=self.frame,
                R_AbarA=desired_pose.rotation(),
                frameBbar=self.plant.world_frame(),
                R_BbarB=RotationMatrix(),
                theta_bound=1e-3,
            )
        prog = ik.prog()
        result = Solve(prog)
        q_desired = result.GetSolution(ik.q())

        # when free block
        # output.set_value(q_desired[:6])
        output.set_value(q_desired[: self.out_port_len])


def add_ground_with_friction(plant, height=0.0):
    dissipation = 5e2
    slab_thickness = 5.0
    hydroelastic_modulus = 5e6

    proximity_properties_ground = ProximityProperties()
    AddContactMaterial(
        dissipation=dissipation,
        friction=CoulombFriction(static_friction=1.0, dynamic_friction=1.0),
        properties=proximity_properties_ground,
    )
    # taken from: https://github.com/vincekurtz/drake_ddp/blob/b4b22a55448121153f992cae453236f7f5891b23/panda_fr3.py#L79
    AddCompliantHydroelasticPropertiesForHalfSpace(
        slab_thickness, hydroelastic_modulus, proximity_properties_ground
    )

    plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform(np.array([0.0, 0.0, height])),
        HalfSpace(),
        "ground_collision",
        proximity_properties_ground,
    )


def add_soft_collisions(
    plant,
    link_name,
    body=Cylinder(radius=0.013, length=0.05),
    collision_pose=RigidTransform(np.array([0.0, 0, 0.0])),
):
    proximity_properties_feet = ProximityProperties()
    AddContactMaterial(
        dissipation=1e0,
        friction=CoulombFriction(static_friction=0.9, dynamic_friction=0.8),
        properties=proximity_properties_feet,
    )
    AddCompliantHydroelasticProperties(
        resolution_hint=5e-2,
        hydroelastic_modulus=5e5,
        properties=proximity_properties_feet,
    )
    coll_id = plant.RegisterCollisionGeometry(
        plant.GetBodyByName(link_name),
        collision_pose,
        body,
        link_name + "_collision",
        proximity_properties_feet,
    )
    return coll_id


def AddRobotModel(
    plant,
    package_path,
    package_name,
    urdf_name,
    scene_graph=None,
    weld_frame_transform=None,
):
    if scene_graph is None:
        parser = Parser(plant, scene_graph)
    else:
        parser = Parser(plant)
    urdf_utils = URDFutils(package_path, package_name, urdf_name)
    urdf_utils.modify_meshes(in_mesh_format=".STL")
    logging.warning("removing collision tags!")
    urdf_utils.remove_collisions_except([])
    # unique_id = urdf_utils.make_model_unique()
    unique_id = ""
    urdf_utils.add_joint_limits()
    urdf_utils.fix_joints_except(["joint_1", "joint_2"])
    urdf_utils.add_actuation_tags()
    _, temp_urdf = urdf_utils.get_modified_urdf()
    # urdf_utils.show_temp_urdf(temp_urdf)
    abs_path = Path(package_path).resolve().__str__()
    parser.package_map().Add(package_name.split("/")[0], abs_path + "/" + package_name)
    robot_model = parser.AddModels(temp_urdf.name)[0]

    if weld_frame_transform is not None:
        plant.WeldFrames(
            plant.get_body(plant.GetBodyIndices(robot_model)[0]).body_frame(),
            plant.world_frame(),
            weld_frame_transform,
        )

    return robot_model, unique_id


def add_env_objects(plant, scene_graph, mesh_scale=1.0):
    parser = Parser(plant, scene_graph)
    urdf_path = (
        (
            Path(__file__).resolve().parent.parent.parent.parent
            / "assets/tblock_paper/scaled_tblock.sdf"
        )
        .resolve()
        .__str__()
    )

    # mesh = trimesh.load_mesh(urdf_path)
    # mesh.apply_scale(mesh_scale)
    # tmp_dir = tempfile.mkdtemp()
    # scaled_mesh_path = Path(tmp_dir) / "scaled_tblock.obj"
    # mesh.export(scaled_mesh_path)
    # then move the mesh and use mesh_to_model from pydrake to get the sdf
    # https://drake.mit.edu/pydrake/pydrake.multibody.mesh_to_model.html?highlight=convert%20sdf

    tblock = parser.AddModels(urdf_path)[0]
    return tblock


def MakeHardwareStation(
    time_step,
    package_path,
    package_name,
    urdf_name,
    weld_frame_transform=None,
    uid=None,
):
    robot_builder = RobotDiagramBuilder(time_step=time_step)
    builder = robot_builder.builder()
    controller_plant = robot_builder.plant()
    scene_graph = robot_builder.scene_graph()
    robot_model, _ = AddRobotModel(
        controller_plant,
        package_path,
        package_name,
        urdf_name,
        weld_frame_transform=weld_frame_transform,
    )
    controller_plant.Finalize()
    control_plant_pos = controller_plant.num_positions(robot_model)
    control_plant_vel = controller_plant.num_velocities(robot_model)
    robot_controller = builder.AddSystem(
        InverseDynamicsController(
            controller_plant,
            kp=[400.0] * control_plant_pos,
            ki=[0.0] * control_plant_pos,
            kd=[40.0] * control_plant_pos,
            has_reference_acceleration=False,
        )
    )
    builder.ExportInput(
        robot_controller.get_input_port_estimated_state(), "robot_estimated_state"
    )
    # builder.ExportOutput(robot_controller.get_output_port(), "robot_torque_commanded")
    builder.ExportOutput(
        robot_controller.GetOutputPort("actuation"), "robot_torque_commanded"
    )
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            control_plant_pos, time_step, suppress_initial_transient=True
        )
    )
    desired_state_from_position.set_name("desired_state_from_position")
    builder.Connect(
        desired_state_from_position.get_output_port(),
        robot_controller.get_input_port_desired_state(),
    )
    builder.ExportInput(
        desired_state_from_position.get_input_port(), "robot_state_desired"
    )
    controller_diagram = builder.Build()
    controller_diagram.set_name("station")
    if uid is not None:
        controller_diagram.set_name("station_" + str(uid))
    return controller_diagram
