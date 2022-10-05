import bpy
from ctypes import ArgumentError
import os

with_temperature = True

tx = -2.0
ty = 4.0
tz = 1.5

rx = 80.0
ry = 0.0
rz = 200.0

fov = 55.0
resolution_x = 800
resolution_y = 800

def clear_non_mesh():
    scene = bpy.context.scene
    #deleteListObjects = ['MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'HAIR', 'POINTCLOUD', 'VOLUME', 'GPENCIL',
    #                 'ARMATURE', 'LATTICE', 'EMPTY', 'LIGHT', 'LIGHT_PROBE', 'CAMERA', 'SPEAKER']
    
    for obj in scene.objects:
        #if obj.type not in ['MESH']:
        if obj.name not in ['CustomObject']:
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
    principled.inputs['Base Color'].default_value = (*tuple(i /255 for i in rgb), 1)
    principled.inputs["Alpha"].default_value = alpha
    
def add_obstacle():
    mesh = bpy.ops.mesh.primitive_cube_add(location=(0, -1.3, 0))
    bpy.ops.transform.resize(value=(10, 3., 0.76))
    obj = bpy.context.active_object
    obj.name = "CustomObstacle"
    
    color = (111, 76, 91)
    set_object_color(obj, color, 0.4)
        
    
def add_background():
    mesh = bpy.ops.mesh.primitive_plane_add(location=(0, 7, 0.5))
    bpy.ops.transform.resize(value=(20, 10, 1))
    
#    bpy.ops.object.mode_set( mode   = 'EDIT'   )
#    bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
    
    bpy.ops.object.mode_set(mode = 'OBJECT')
    obj = bpy.context.active_object
    obj.name = "CustomPlane"
    
    color = (253, 251, 246)
    set_object_color(obj, color, 1)
    
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    
    bpy.ops.object.mode_set(mode = 'OBJECT')
    obj.data.edges[1].select = True
    bpy.ops.object.mode_set(mode = 'EDIT') 

    bpy.ops.mesh.extrude_edges_move(
        TRANSFORM_OT_translate={"value":(0, 0, 10)})

    bpy.ops.object.mode_set(mode = 'OBJECT')
    
def set_camera():
    scene = bpy.context.scene

    camera_data = bpy.data.cameras.new(name='CustomCameraData')
    scene.camera = bpy.data.objects.new('CustomCamera', camera_data)
    bpy.context.collection.objects.link(scene.camera)
    #bpy.context.scene.collection.objects.link(scene.camera)

    # Set camera fov in degrees
    pi = 3.14159265
    scene.camera.data.angle = fov*(pi/180.0)

    # Set camera rotation in euler angles
    scene.camera.rotation_mode = 'XYZ'
    scene.camera.rotation_euler[0] = rx*(pi/180.0)
    scene.camera.rotation_euler[1] = ry*(pi/180.0)
    scene.camera.rotation_euler[2] = rz*(pi/180.0)

    # Set camera translation
    scene.camera.location = (tx, ty, tz)


def add_light():
    light_data = bpy.data.lights.new(name="CustomLightData", type='POINT')
    light_data.energy = 5500
    light_data.shadow_soft_size = 1

    light_object = bpy.data.objects.new(name="CustomLight", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    s = 2
    light_object.location = (tx * s, ty * s, tz * s) 
    dg = bpy.context.evaluated_depsgraph_get() 
    dg.update()

    #bpy.context.object.data.cycles.cast_shadow = False
    bpy.data.objects["CustomObject"].visible_shadow = True
    bpy.data.objects["CustomObstacle"].visible_shadow = True #False


def set_object_material():
    obj = bpy.data.objects["CustomObject"]
    mat = crete_material("CustomObjectMaterial", obj)
    
    tree = mat.node_tree
    principled = tree.nodes["Principled BSDF"]
    principled.inputs["Specular"].default_value = 0.3


    color_node = tree.nodes.new("ShaderNodeVertexColor")
    if with_temperature:
        tree.links.new(color_node.outputs["Color"], principled.inputs["Base Color"])
    else:
        principled.inputs['Base Color'].default_value = (0.2,0.2,0.2,1)
        


def set_world():
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0
    bpy.context.scene.world.color = (1,1,1)
    
def set_workbench():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.color_type = 'VERTEX'
    scene.display.shading.use_world_space_lighting = True
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.studiolight_rotate_z = 0
    
def set_cycles():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.view_settings.look = 'High Contrast'
    
    scene.cycles.use_preview_denoising = True
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPTIX'
    bpy.context.scene.cycles.preview_denoiser = 'OPTIX'

    scene.cycles.use_fast_gi = True
    scene.world.light_settings.ao_factor = 0.4
    scene.world.light_settings.distance = 0.1
    
def set_render():
    scene = bpy.context.scene
    scene.render.fps_base = 3

    # Set render resolution
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y

    # Set output type
    if True:
        scene.render.filepath = '../result'
        bpy.context.scene.render.image_settings.file_format = 'PNG'
    else:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        bpy.context.scene.render.ffmpeg.constant_rate_factor = 'LOSSLESS'

    if True:
        bpy.ops.render.render(animation=True)



    
clear_non_mesh()
clear_materials()
add_background() 
add_obstacle()
set_camera()
add_light()
set_object_material()

set_world()
set_cycles()
set_render()
