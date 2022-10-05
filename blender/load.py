import os
import pickle
from io import BufferedReader
from random import random
import bpy
import numpy
from contextlib import redirect_stdout
import io
from bpy import context
import builtins as __builtin__
from time import time
import bmesh
from random import uniform
from ctypes import ArgumentError

import matplotlib
import matplotlib.pyplot as plt
import os


directory = "../../conmech/output" #"../scenarios"
dense = True

def find_files_by_extension(directory, extension):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(f".{extension}")]:
            path = os.path.join(dirpath, filename)
            files.append(path)
    return files

scene_files = find_files_by_extension(directory, "scenes_data")
path_id = '/scenarios/' if dense else '/scenarios_reduced/'
scene_files = [f for f in scene_files if path_id in f]
scene_files.sort()
file_name = scene_files[-1]
all_arrays_path = f"{directory}/{file_name}"

#raise ArgumentError(file_name)

def insert_keyframe(fcurves, frame, values):
    for fcu, val in zip(fcurves, values):
        fcu.keyframe_points.insert(frame, val, options={'FAST'})

def main():
    clear_scene()
    simulation, with_temperature = load_simulation()
    initial_nodes, initial_elements = simulation[0][:2]
    mesh, object = create_mesh(initial_nodes, initial_elements)
    
    #obj = bpy.context.active_object
    #mesh = obj.data
    action = bpy.data.actions.new("MeshAnimation")

    mesh.animation_data_create()
    mesh.animation_data.action = action

    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    color_layer = mesh.vertex_colors.active  
    
    skip = 1
    frame_num = 0
    steps = len(simulation)
    bpy.context.scene.frame_end = steps

    for v in mesh.vertices:
        fcurves = [action.fcurves.new("vertices[%d].co" % v.index, index = i) for i in range(3)]

        for frame_num in range(0, steps, skip):
            nodes, elements = simulation[frame_num][:2]
            co_kf = nodes[v.index]
            insert_keyframe(fcurves, frame_num, co_kf)

    object.select_set(True)
    bpy.context.view_layer.objects.active = object
    #bpy.ops.paint.vertex_paint_toggle()
    
    if not with_temperature:
        return
    
    norm_max_t = 0.005 #1 #2 #1
    norm_min_t = -0.0009 #0.1 
    def normalize_t(t):
        return (t - norm_min_t) / (norm_max_t - norm_min_t)

    cmap = plt.cm.plasma #coolwarm #magma #cool
    norm = matplotlib.colors.Normalize(vmin=0., vmax=1.)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    def get_color(temp):
        return mapper.to_rgba(normalize_t(temp))[0] # numpy.array([temp,0,0,1])
    
    max_t = 0
    min_t = 0
    index = 0
    for poly in mesh.polygons:
        for idx in poly.vertices:
            fcurves_color = [action.fcurves.new("vertex_colors.active.data[%d].color" % index, index = i) for i in range(4)]

            for frame_num in range(0, steps, skip):
                nodes, elements, temperatures = simulation[frame_num]
                node = nodes[idx]
                temp = temperatures[idx]
                if temp > max_t:
                    max_t = temp
                if temp < min_t:
                    min_t = temp
                # RGB, 0 = dark, 1 = light
                color = get_color(temp) #numpy.array([temp,0,0,1]) #(node[2])
                insert_keyframe(fcurves_color, frame_num, color)
            index += 1       
            
    #raise ArgumentError(max_t)  
    #raise ArgumentError(min_t)     
            

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
    return simulation, with_temperature


def clear_scene():
    scene = bpy.context.scene

    for obj in scene.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
        #obj.select_set(True)

    bpy.context.evaluated_depsgraph_get().update()
    #if bpy.context.selected_objects:
    #    bpy.ops.object.delete() 
    bpy.ops.outliner.orphans_purge()

def create_mesh(nodes, elements):
    mesh = bpy.data.meshes.new(name="CustomMesh")
    mesh.from_pydata(nodes, [], elements)
    mesh.update(calc_edges=True)
    mesh.validate()

    object = bpy.data.objects.new("CustomObject", mesh)
    bpy.context.collection.objects.link(object)
    #bpy.context.scene.collection.objects.link(scene.camera)

    return mesh, object

    
if __name__ == "__main__": 
    print("START")
    start = time()
    main()
    print(time() - start)
    print("DONE")
