from pydrake.all import (
    MultibodyPlant,
    Parser,
    RigidTransform,
    RotationMatrix,
    LeafSystem,
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
    RobotDiagramBuilder,
    InverseDynamicsController,
    StateInterpolatorWithDiscreteDerivative,
    Frame,
    ContactModel,
    CollisionFilterDeclaration,
    GeometrySet,
)
import numpy as np
from sak.URDFutils import URDFutils
from pathlib import Path
import logging


class PoseToConfig(LeafSystem):
    def __init__(self, plant: MultibodyPlant, frame: Frame, num_dof: int):
        LeafSystem.__init__(self)
        self.plant = plant
        self.frame = frame
        self.plant_context = plant.CreateDefaultContext()
        self.DeclareAbstractInputPort(
            "pose",
            AbstractValue.Make(RigidTransform()),
        )
        self.out_port_len = num_dof
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
        output.set_value(q_desired[: self.out_port_len])


def add_ground_with_friction(plant):
    dissipation = 5e2
    slab_thickness = 5.0
    hydroelastic_modulus = 5e6

    proximity_properties_ground = ProximityProperties()
    AddContactMaterial(
        dissipation=dissipation,
        friction=CoulombFriction(static_friction=1.0, dynamic_friction=1.0),
        properties=proximity_properties_ground,
    )
    AddCompliantHydroelasticPropertiesForHalfSpace(
        slab_thickness, hydroelastic_modulus, proximity_properties_ground
    )
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform(),
        HalfSpace(),
        "ground_collision",
        proximity_properties_ground,
    )


def add_soft_collisions(plant, eef_link_name):
    dissipation = 1e4
    point_stiffness = 1e7
    surface_friction_feet = CoulombFriction(static_friction=0, dynamic_friction=0)
    proximity_properties_feet = ProximityProperties()
    AddContactMaterial(
        dissipation, point_stiffness, surface_friction_feet, proximity_properties_feet
    )
    AddCompliantHydroelasticProperties(0.05, 5e6, proximity_properties_feet)

    radius, length = 0.013, 0.05
    offset = np.array([0.0, 0, 0.19])
    plant.RegisterCollisionGeometry(
        plant.GetBodyByName(eef_link_name),
        RigidTransform(offset),
        Cylinder(radius=radius, length=length),
        eef_link_name + "_collision",
        proximity_properties_feet,
    )


def AddRobotModel(
    plant,
    package_path,
    package_name,
    urdf_name,
    scene_graph=None,
    weld_frame_transform=RigidTransform(),
):
    if scene_graph is None:
        parser = Parser(plant, scene_graph)
    else:
        parser = Parser(plant)
    urdf_utils = URDFutils(package_path, package_name, urdf_name)
    urdf_utils.modify_meshes(in_mesh_format=".STL")
    logging.warning("removing collision tags!")
    urdf_utils.remove_collisions_except([])
    unique_id = ""
    urdf_utils.add_joint_limits()
    urdf_utils.add_actuation_tags()
    _, temp_urdf = urdf_utils.get_modified_urdf()
    abs_path = Path(package_path).resolve().__str__()
    parser.package_map().Add(package_name.split("/")[0], abs_path + "/" + package_name)
    robot_model = parser.AddModels(temp_urdf.name)[0]
    try:
        weld_frame = plant.WeldFrames(
            plant.get_body(plant.GetBodyIndices(robot_model)[0]).body_frame(),
            plant.world_frame(),
            weld_frame_transform,
        )
    except RuntimeError as e:
        logging.error(
            f"Failed to weld frame for {robot_model} with error: {e}. "
            "Ensure the weld_frame_transform is correct."
        )

    return robot_model, unique_id


def configure_contacts(plant, eef_link_name, scene_graph, robot_model_instance):
    plant.set_contact_model(ContactModel.kHydroelasticsOnly)
    add_ground_with_friction(plant)
    add_soft_collisions(plant, eef_link_name=eef_link_name)
    plant.set_penetration_allowance(1e-5)
    collision_filter_manager = scene_graph.collision_filter_manager()
    collision_filter_manager.Apply(
        CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(
                plant.GetCollisionGeometriesForBody(
                    plant.GetBodyByName(
                        eef_link_name,
                        robot_model_instance,
                    )
                )
            ),
            GeometrySet(plant.GetCollisionGeometriesForBody(plant.world_body())),
        )
    )


def add_env_objects(plant, scene_graph):
    parser = Parser(plant, scene_graph)
    urdf_path = (
        (
            Path(__file__).resolve().parent.parent.parent.parent
            / "assets/tblock_paper/tblock_paper.sdf"
        )
        .resolve()
        .__str__()
    )
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
            kp=[100.0] * control_plant_pos,
            ki=[0.0] * control_plant_pos,
            kd=[20.0] * control_plant_pos,
            has_reference_acceleration=False,
        )
    )
    builder.ExportInput(
        robot_controller.get_input_port_estimated_state(), "robot_estimated_state"
    )
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
