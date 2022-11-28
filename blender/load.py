import sys

# import pip
# pip.main(['install', 'tqdm', '--user'])
packages_path = "/home/michal/.local/lib/python3.10/site-packages"
sys.path.insert(0, packages_path)
# print(sys.path)

import builtins as __builtin__
import io
import os
import pickle
from ctypes import ArgumentError
from io import BufferedReader
from random import random, uniform
from time import time

import bmesh
import bpy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

input_path = "/home/michal/Desktop/conmech/output"
dense = True

load_mesh = True
load_world = True
render = True
draw_obstacle = True

cycles = True  # False
output_video = True  # True
output_path = "/home/michal/Desktop/conmech/output"


def get_tqdm(iterable, desc=None, position=None) -> tqdm:
    return tqdm(iterable, desc=desc, position=position, ascii=True)


def find_files_by_extension(directory, extension):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(f".{extension}")]:
            path = os.path.join(dirpath, filename)
            files.append(path)
    return files


scene_files = find_files_by_extension(input_path, "scenes_data")
path_id = "/scenarios/" if dense else "/scenarios_reduced/"
scene_files = [f for f in scene_files if path_id in f]
all_arrays_path = max(scene_files, key=os.path.getctime)
all_arrays_name = os.path.basename(all_arrays_path).split("DATA")[0]

print(f"FILE: {all_arrays_name}")
# raise ArgumentError(file_name)


def load_data():
    simulation, temperature = load_simulation()
    with_temperature = temperature is not None

    initial_nodes, initial_elements = simulation[0][:2]
    mesh, object = create_mesh(initial_nodes, initial_elements, with_temperature)

    # obj = bpy.context.active_object
    # mesh = obj.data
    action = bpy.data.actions.new("MeshAnimation")

    mesh.animation_data_create()
    mesh.animation_data.action = action

    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    # color_layer = mesh.vertex_colors.active

    steps = len(simulation)
    frames = range(steps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = steps

    # for v in get_tqdm(mesh.vertices, "Vertices"):
    #     fcurves = [action.fcurves.new(f"vertices[{v.index}].co", index=i) for i in range(3)]

    #     for frame_num in frames:
    #         nodes, elements = simulation[frame_num][:2]
    #         node = nodes[v.index]
    #         # insert_keyframe(fcurves, frame_num, node)
    #         for fcu, val in zip(fcurves, node):
    #             fcu.keyframe_points.insert(frame_num, val, options={"FAST"})

    #     for fc in fcurves:
    #         fc.update()

    for v in get_tqdm(mesh.vertices, "Vertices"):
        for i in range(3):
            samples = [simulation[frame_num][0][v.index][i] for frame_num in frames]
            fc = action.fcurves.new(f"vertices[{v.index}].co", index=i)
            fc.keyframe_points.add(count=len(frames))
            fc.keyframe_points.foreach_set("co", [x for co in zip(frames, samples) for x in co])
            fc.update()

    object.select_set(True)
    bpy.context.view_layer.objects.active = object
    # bpy.ops.paint.vertex_paint_toggle()

    if not with_temperature:
        print("WITHOUT TEMPERATURE")
        return

    print("WITH TEMPERATURE")
    max_t = np.max(temperature)
    min_t = np.min(temperature)
    print(f"MAX_TEMP: {max_t}")
    print(f"MIN_TEMP: {min_t}")
    # exit()
    # raise ArgumentError(max_t)
    # raise ArgumentError(min_t)

    norm_max_t = 3.0  # 0.005
    norm_min_t = 0.0  # -0.0009

    def normalize_t(t):
        return (t - norm_min_t) / (norm_max_t - norm_min_t)

    cmap = plt.cm.plasma  # coolwarm #magma #cool
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    def get_color(temp):
        return np.array(mapper.to_rgba(normalize_t(temp)))  # np.array([temp,0,0,1])

    # TODO: make better
    all_colors = np.array(
        [
            [get_color(temperature[frame_num, idx]) for frame_num in frames]
            for idx in range(len(mesh.vertices))
        ]
    )
    index = 0
    for poly in get_tqdm(mesh.polygons, "Temperature - polygons"):
        for idx in poly.vertices:
            for i in range(4):
                samples = all_colors[idx, :, i]
                fc = action.fcurves.new(f"vertex_colors.active.data[{index}].color", index=i)
                fc.keyframe_points.add(count=len(frames))
                fc.keyframe_points.foreach_set("co", [x for co in zip(frames, samples) for x in co])
                fc.update()
            index += 1

    # for poly in get_tqdm(mesh.polygons, "Temperature - polygons"):
    #     for idx in poly.vertices:
    #         fcurves_color = [
    #             action.fcurves.new(f"vertex_colors.active.data[{index}].color", index=i)
    #             for i in range(4)
    #         ]

    #         for frame_num in range(0, steps, skip):
    #             temp = temperature[frame_num, idx]
    #             if temp > max_t:
    #                 max_t = temp
    #             if temp < min_t:
    #                 min_t = temp
    #             # RGB, 0 = dark, 1 = light
    #             color = get_color(temp)  # np.array([temp,0,0,1]) #(node[2])
    #             for fc, val in zip(fcurves_color, color):
    #                 fc.keyframe_points.insert(frame_num, val, options={"FAST"})

    #         index += 1

    #         for fc in fcurves_color:
    #             fc.update()


def get_all_indices(data_path):
    all_indices = []
    try:
        with open(f"{data_path}_indices", "rb") as file:
            try:
                while True:
                    all_indices.append(pickle.load(file))
            except EOFError:
                pass
    except IOError:
        pass
    return all_indices


def open_file_read(path: str):
    return open(path, "rb")


def load_byte_index(byte_index: int, data_file: BufferedReader):
    data_file.seek(byte_index)
    data = pickle.load(data_file)
    return data


def load_simulation():
    all_indices = get_all_indices(all_arrays_path)
    simulation = []
    temperature = None
    scenes_file = open_file_read(all_arrays_path)
    with scenes_file:
        for step in range(len(all_indices)):
            byte_index = all_indices[step]
            arrays = load_byte_index(
                byte_index=byte_index,
                data_file=scenes_file,
            )
            simulation.append(arrays)
            with_temperature = len(arrays) > 2
            if with_temperature:
                if temperature is None:
                    temperature = []
                temperature.append(arrays[2])

    if with_temperature:
        temperature = np.array(temperature)[..., 0]
    return simulation, temperature


def clear_scene():
    scene = bpy.context.scene

    for obj in scene.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
        # obj.select_set(True)

    bpy.context.evaluated_depsgraph_get().update()
    # if bpy.context.selected_objects:
    #    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()


def object_with_temperature():
    if bpy.context.scene.objects.get("CustomObjectTemperature"):
        return True
    if bpy.context.scene.objects.get("CustomObject"):
        return False
    raise ArgumentError()
    # mesh_fcurves = bpy.data.actions["MeshAnimation"].fcurves
    # return mesh_fcurves.find("vertex_colors.active.data[0].color") is not None


def get_object_name(with_temperature=None):
    if with_temperature is None:
        with_temperature = object_with_temperature()
    return "CustomObjectTemperature" if with_temperature else "CustomObject"


def create_mesh(nodes, elements, with_temperature):
    mesh = bpy.data.meshes.new(name="CustomMesh")
    mesh.from_pydata(nodes, [], elements)
    mesh.update(calc_edges=True)
    mesh.validate()

    object_name = get_object_name(with_temperature)
    object = bpy.data.objects.new(object_name, mesh)
    bpy.context.collection.objects.link(object)
    # bpy.context.scene.collection.objects.link(scene.camera)

    return mesh, object


######################################################################


def clear_non_mesh():
    scene = bpy.context.scene
    # deleteListObjects = ['MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'HAIR', 'POINTCLOUD', 'VOLUME', 'GPENCIL',
    #                 'ARMATURE', 'LATTICE', 'EMPTY', 'LIGHT', 'LIGHT_PROBE', 'CAMERA', 'SPEAKER']

    for obj in scene.objects:
        # if obj.type not in ['MESH']:
        if "CustomObject" not in obj.name:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.context.evaluated_depsgraph_get().update()
    bpy.ops.outliner.orphans_purge()


def clear_materials():
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)


def crete_material(name, obj):
    obj.data.materials.clear()
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    obj.data.materials.append(mat)
    obj.active_material_index = 0
    return mat


def set_object_color(obj, rgb, alpha):
    mat = crete_material(f"{obj.name}Material", obj)
    tree = mat.node_tree
    principled = tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (*tuple(i / 255 for i in rgb), 1)
    principled.inputs["Alpha"].default_value = alpha


def add_obstacle():
    if not draw_obstacle or not object_with_temperature():
        return

    mesh = bpy.ops.mesh.primitive_cube_add(location=(0, -1.3, 0))
    bpy.ops.transform.resize(value=(10, 3.0, 0.76))
    obj = bpy.context.active_object
    obj.name = "CustomObstacle"

    color = (111, 76, 91)
    set_object_color(obj, color, 0.4)


def add_background():
    location = (0, 7, 0.5) if object_with_temperature() else (4, 2, -1)

    mesh = bpy.ops.mesh.primitive_plane_add(location=location)
    bpy.ops.transform.resize(value=(20, 10, 1))

    #    bpy.ops.object.mode_set( mode   = 'EDIT'   )
    #    bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')

    bpy.ops.object.mode_set(mode="OBJECT")
    obj = bpy.context.active_object
    obj.name = "CustomPlane"

    color = (253, 251, 246)
    set_object_color(obj, color, 1)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action="DESELECT")

    bpy.ops.object.mode_set(mode="OBJECT")
    for edge in obj.data.edges:
        edge.select = True
    bpy.ops.object.mode_set(mode="EDIT")

    bpy.ops.mesh.extrude_edges_move(TRANSFORM_OT_translate={"value": (0, 0, 10)})

    bpy.ops.object.mode_set(mode="OBJECT")


def get_camera_location():
    if object_with_temperature():
        return (-2.0, 4.0, 1.5)
    return (-3.0, 1.0, 0.2)


def get_camera_rotation():
    if object_with_temperature():
        return (80.0, 0.0, 200.0)
    return (90.0, 0.0, 270.0)


def get_resolution():
    if object_with_temperature():
        return (800, 800)
    return (1200, 1200)


def set_camera():
    rx, ry, rz = get_camera_rotation()

    fov = 55.0

    scene = bpy.context.scene

    camera_data = bpy.data.cameras.new(name="CustomCameraData")
    scene.camera = bpy.data.objects.new("CustomCamera", camera_data)
    bpy.context.collection.objects.link(scene.camera)
    # bpy.context.scene.collection.objects.link(scene.camera)

    # Set camera fov in degrees
    pi = 3.14159265
    scene.camera.data.angle = fov * (pi / 180.0)

    # Set camera rotation in euler angles
    scene.camera.rotation_mode = "XYZ"
    scene.camera.rotation_euler[0] = rx * (pi / 180.0)
    scene.camera.rotation_euler[1] = ry * (pi / 180.0)
    scene.camera.rotation_euler[2] = rz * (pi / 180.0)

    # Set camera translation
    scene.camera.location = get_camera_location()


def add_light():
    light_data = bpy.data.lights.new(name="CustomLightData", type="POINT")
    light_data.energy = 5500
    light_data.shadow_soft_size = 1

    light_object = bpy.data.objects.new(name="CustomLight", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    s = 2
    light_object.location = tuple(s * np.array(get_camera_location()))
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    # bpy.context.object.data.cycles.cast_shadow = False
    for obj in bpy.data.objects:
        obj.visible_shadow = True
    # with_temperature = object_with_temperature()
    # object_name = get_object_name(with_temperature)
    # bpy.data.objects[object_name].visible_shadow = True
    # bpy.data.objects["CustomObstacle"].visible_shadow = True  # False


def set_object_material():
    object_name = get_object_name()
    obj = bpy.data.objects[object_name]
    mat = crete_material("CustomObjectMaterial", obj)

    tree = mat.node_tree
    principled = tree.nodes["Principled BSDF"]
    principled.inputs["Specular"].default_value = 0.3

    color_node = tree.nodes.new("ShaderNodeVertexColor")
    if object_with_temperature():
        tree.links.new(color_node.outputs["Color"], principled.inputs["Base Color"])
    else:
        principled.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1)


def set_world():
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0
    bpy.context.scene.world.color = (1, 1, 1)


def set_workbench():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"

    scene.display.shading.color_type = "VERTEX"
    scene.display.shading.use_world_space_lighting = True
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studiolight_rotate_z = 0


def set_cycles():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    # enable_gpus("CUDA")
    scene.view_settings.look = "High Contrast"

    scene.cycles.use_preview_denoising = True
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = "OPTIX"
    bpy.context.scene.cycles.preview_denoiser = "OPTIX"

    scene.cycles.use_fast_gi = True
    scene.world.light_settings.ao_factor = 0.4
    scene.world.light_settings.distance = 0.1


def enable_gpus(device_type):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()

    if device_type == "CUDA":
        devices = cuda_devices
    elif device_type == "OPENCL":
        devices = opencl_devices
    else:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []

    for device in devices:
        if device.type == "CPU":
            device.use = False
        else:
            device.use = True
            activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus

def set_render():
    if cycles:
        set_cycles()
    else:
        set_workbench()

    scene = bpy.context.scene
    scene.render.fps_base = 3

    # Set render resolution
    r_x, r_y = get_resolution()
    scene.render.resolution_x = r_x
    scene.render.resolution_y = r_y

    scene.render.filepath = f"{output_path}/{time()}_{all_arrays_name}_"

    # Set output type
    if not output_video:
        bpy.context.scene.render.image_settings.file_format = "PNG"
    else:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        bpy.context.scene.render.ffmpeg.constant_rate_factor = "LOSSLESS"


def render_animation():
    if not output_video:
        scene = bpy.context.scene
        fp = scene.render.filepath
        scene.render.image_settings.file_format = "PNG"  # set output format to .png

        skip = 10
        frames = range(0, bpy.context.scene.frame_end + 1, skip)

        for frame_nr in frames:
            scene.frame_set(frame_nr)
            scene.render.filepath = fp + str(frame_nr)
            bpy.ops.render.render(write_still=True)

        scene.render.filepath = fp
    else:
        bpy.ops.render.render(animation=True)


def set_scene_and_render():
    clear_non_mesh()
    clear_materials()
    add_background()
    add_obstacle()
    set_camera()
    add_light()
    set_object_material()

    set_world()
    set_render()


if __name__ == "__main__":
    print("---START---")
    start = time()
    if load_mesh:
        print("---LOADING MESH---")
        clear_scene()
        load_data()

    if load_world:
        print("---LOADING WORLD---")
        set_scene_and_render()

    if render:
        print("---RENDERING---")
        render_animation()
    print(f"TIME: {time() - start}")
    print("---DONE---")
