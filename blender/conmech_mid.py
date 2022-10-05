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


def insert_keyframe(fcurves, frame, values):
    for fcu, val in zip(fcurves, values):
        fcu.keyframe_points.insert(frame, val, options={'FAST'})



def color_mesh(mesh):
    
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()

    #color_layer = mesh.vertex_colors["Col"]
    color_layer = mesh.vertex_colors.active  

    i = 0
    print(len(color_layer.data))
    for poly in mesh.polygons:
        #mean = np.zeros_like(mesh.polygons)
        #for idx in poly.loop_indices:
        #    mean += (mesh.vertices[idx] / len(poly.loop_indices))
            
        r, g, b = [random() for i in range(3)]
        #r = mean[0]
        for idx in poly.loop_indices:
            color_layer.data[i].color = (r,0,0,1)#(r, g, b, 1.0)
            color_layer.data[i].keyframe_insert("color", index=-1, frame=frame_num)
            i += 1

    #bpy.ops.object.mode_set(mode='VERTEX_PAINT')
    
    
def main():
    prepare_scene()
    
    simulation = load_simulation()
    initial_nodes, initial_elements = simulation[0]
    mesh = create_mesh(initial_nodes, initial_elements)
    
    #obj = bpy.context.active_object
    #mesh = obj.data
    action = bpy.data.actions.new("MeshAnimation")

    mesh.animation_data_create()
    mesh.animation_data.action = action

    data_path = "vertices[%d].co"
   
    skip = 5
    frame_num = 0
    steps = len(simulation)
    bpy.context.scene.frame_end = steps

    for v in mesh.vertices:
        fcurves = [action.fcurves.new(data_path % v.index, index = i) for i in range(3)]

        for frame_num in range(0, steps, skip):
            nodes, elements = simulation[frame_num]
            co_kf = nodes[v.index]
            insert_keyframe(fcurves, frame_num, co_kf)




#def console_print(*args, **kwargs):
#    for a in context.screen.areas:
#        if a.type == 'CONSOLE':
#            c = {}
#            c['area'] = a
#            c['space_data'] = a.spaces.active
#            c['region'] = a.regions[-1]
#            c['window'] = context.window
#            c['screen'] = context.screen
#            s = " ".join([str(arg) for arg in args])
#            for line in s.split("\n"):
#                bpy.ops.console.scrollback_append(c, text=line)
               
#def print(*args, **kwargs):
#    console_print(*args, **kwargs)       # to Python Console
#    __builtin__.print(*args, **kwargs)   # to System Console
#    
    
        
main_path = "/home/michal/Desktop/Blender/scenarios"
        
#all_arrays_path = "C:/Users/michal/Desktop/Blender/scenarios/bunny_fall_DATA.scenes_data"
#all_arrays_path = "C:/Users/michal/Desktop/Blender/scenarios/bunny_roll_DATA.scenes_data"
#all_arrays_path = "C:/Users/michal/Desktop/Blender/scenarios/twist_roll_DATA.scenes_data"
#all_arrays_path = f"{main_path}/ball_throw_DATA.scenes_data"
#all_arrays_path = f"{main_path}/ball_rotate_DATA.scenes_data"
#all_arrays_path = f"{main_path}/bunny_rotate_DATA.scenes_data"
#all_arrays_path = f"{main_path}/bunny_swing_DATA.scenes_data"
all_arrays_path = f"{main_path}/bunny_fall_DATA.scenes_data"
#all_arrays_path = f"{main_path}/temperature_3d_bunny_throw_DATA.scenes_data"

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
            nodes, elements = arrays
            nodes = nodes #* 100
            simulation.append((nodes, elements))
    return simulation


def prepare_scene():
    scene = bpy.context.scene

    for obj in scene.collection.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    if False:
        cam1 = bpy.data.cameras.new("Camera 1")
        cam1.lens = 18

        cam_obj1 = bpy.data.objects.new("Camera 1", cam1)
        cam_obj1.location = (2, -3, 3)
        cam_obj1.rotation_euler = (1.2, 0.1, 0.3)
        scene.collection.objects.link(cam_obj1)

        scene.camera = cam_obj1

        light_data = bpy.data.lights.new(name="Light", type="SUN")
        light_data.energy = 300
        light_data.color = (0, 0, 1)

        light_object = bpy.data.objects.new(name="Light", object_data=light_data)

        bpy.context.collection.objects.link(light_object)

        bpy.context.view_layer.objects.active = light_object

        light_object.location = (5, 5, 5)

    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    # bpy.data.collections.remove(scn.collection)


def create_mesh(nodes, elements):
    mesh = bpy.data.meshes.new(name="ConmechMesh")
    mesh.from_pydata(nodes, [], elements)
    mesh.update(calc_edges=True)
    mesh.validate()

    obj = bpy.data.objects.new("Conmech", mesh)

    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    return mesh


def main2():
    #return 
    prepare_scene()
    simulation = load_simulation()
    initial_nodes, initial_elements = simulation[0]
    mesh = create_mesh(initial_nodes, initial_elements)
        
    skip = 5
    frame_num = 0
    steps = len(simulation)
    bpy.context.scene.frame_end = steps

    for frame_num in range(0, steps, skip):
        print(f"Frame: {frame_num}/{steps}")
        nodes, elements = simulation[frame_num]
        
        mesh.vertices.foreach_set("co", nodes.reshape(-1))
                    
        # mesh.update(calc_edges=True)

        for v in mesh.vertices:
            v.keyframe_insert("co", index=-1, frame=frame_num)


    # bpy.context.scene.render.image_settings.file_format= "AVI_RAW"
    # bpy.context.scene.render.filepath = os.path.join("C:/tmp", "1234")
    # bpy.ops.render.render(animation = True)

    # print("DONE2")




if __name__ == "__main__":
    print("START")
    start = time()
    main2()
    print(time() - start)
    print("DONE")
