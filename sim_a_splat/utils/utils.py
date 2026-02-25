"""Utility functions for the sim_a_splat package."""

import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh


logger = logging.getLogger(__name__)

# Mesh formats supported for conversion to .obj
MESH_EXTENSIONS = {".stl", ".dae", ".ply", ".glb", ".gltf", ".fbx", ".collada"}


def convert_meshes_to_obj(robot_description_path):
    """
    Recursively find all mesh files under any meshes/ directory within
    robot_description_path and convert them to .obj format in-place
    (same filename, .obj extension).

    Args:
        robot_description_path (str or Path): Root path of the robot description.

    Returns:
        list[Path]: Paths to the .obj files that were written.
    """
    robot_description_path = Path(robot_description_path)
    if not robot_description_path.exists():
        raise ValueError(f"Path does not exist: {robot_description_path}")

    converted = []

    for mesh_file in sorted(robot_description_path.rglob("*")):
        # Only process files inside a meshes/ directory
        if "meshes" not in mesh_file.parts:
            continue
        if not mesh_file.is_file():
            continue
        if mesh_file.suffix.lower() not in MESH_EXTENSIONS:
            continue

        obj_path = mesh_file.with_suffix(".obj")
        if obj_path.exists():
            logger.debug(f"Already exists, skipping: {obj_path}")
            continue

        try:
            loaded = trimesh.load(str(mesh_file))
            if isinstance(loaded, trimesh.Scene):
                if loaded.is_empty:
                    logger.warning(f"Empty scene, skipping: {mesh_file}")
                    continue
                # dump(concatenate=True) applies all node transforms (including
                # unit-scale matrices) before concatenating — without this the
                # raw vertex coordinates are exported and end up 1000x too large
                # for DAE files whose geometry is stored in millimetres.
                loaded = loaded.dump(concatenate=True)
            loaded.export(str(obj_path))
            logger.info(f"Converted: {mesh_file.name} -> {obj_path.name}")
            converted.append(obj_path)
        except Exception as e:
            logger.error(f"Failed to convert {mesh_file}: {e}")

    return converted


def convert_urdf_to_drake(urdf_path):
    """
    Read a URDF file and rewrite all mesh filename attributes to use
    package:// style paths pointing to .obj files. Saves the result as
    <original_name>_drake.urdf alongside the original.

    The package name is inferred from the directory that contains the urdf/
    folder (standard ROS layout: <package_root>/urdf/<file>.urdf).

    Args:
        urdf_path (str or Path): Path to the URDF file.

    Returns:
        Path: Path to the written _drake URDF file.
    """
    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.exists():
        raise ValueError(f"URDF does not exist: {urdf_path}")

    # Infer package root: the folder containing urdf/
    package_root = urdf_path.parent.parent
    package_name = package_root.name

    logger.info(f"Package root: {package_root}")
    logger.info(f"Package name: {package_name}")

    ET.register_namespace("", "")
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if not filename:
            continue

        # Resolve to absolute path
        if filename.startswith("package://"):
            pkg, _, rel = filename[len("package://") :].partition("/")
            abs_mesh = (package_root.parent / pkg / rel).resolve()
        else:
            abs_mesh = (urdf_path.parent / filename).resolve()

        abs_obj = abs_mesh.with_suffix(".obj")

        try:
            rel_to_pkg = abs_obj.relative_to(package_root)
            new_filename = f"package://{package_name}/{rel_to_pkg.as_posix()}"
        except ValueError:
            logger.warning(
                f"Mesh {abs_obj} is outside package root {package_root}; "
                f"keeping path with .obj suffix"
            )
            new_filename = str(Path(filename).with_suffix(".obj"))

        logger.debug(f"  {filename!r} -> {new_filename!r}")
        mesh.set("filename", new_filename)

    ET.indent(root)
    output_path = urdf_path.with_name(urdf_path.stem + "_drake.urdf")
    tree.write(str(output_path), xml_declaration=True, encoding="unicode")
    logger.info(f"Written: {output_path}")
    return output_path


def main():
    """Command-line interface: convert meshes and produce a Drake-ready URDF."""
    parser = argparse.ArgumentParser(
        description="Convert robot meshes to .obj and produce a Drake-ready URDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all meshes and rewrite the URDF
  python -m sim_a_splat.utils.utils \\
    robot_description/franka_dual_arm \\
    robot_description/franka_dual_arm/urdf/fr3_duo.urdf

  # Only rewrite the URDF (skip mesh conversion)
  python -m sim_a_splat.utils.utils \\
    robot_description/franka_dual_arm \\
    robot_description/franka_dual_arm/urdf/fr3_duo.urdf \\
    --no-convert-meshes
        """,
    )
    parser.add_argument(
        "robot_description_path",
        type=str,
        help="Path to the robot description package directory (contains meshes/)",
    )
    parser.add_argument(
        "urdf_path",
        type=str,
        help="Path to the URDF file to rewrite",
    )
    parser.add_argument(
        "--no-convert-meshes",
        action="store_true",
        help="Skip mesh conversion, only rewrite the URDF",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    try:
        if not args.no_convert_meshes:
            logger.info("Converting meshes to .obj...")
            converted = convert_meshes_to_obj(args.robot_description_path)
            logger.info(f"Converted {len(converted)} mesh(es)")

        logger.info("Rewriting URDF for Drake...")
        drake_urdf = convert_urdf_to_drake(args.urdf_path)
        logger.info(f"Drake URDF: {drake_urdf}")
        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
