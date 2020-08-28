import bpy
import numpy as np
from scipy.spatial.transform import Rotation


def create_axis(name, rot, tvec):
    bpy.ops.mesh.primitive_cone_add(
        radius1=0.2, depth=0.4, location=(0, 0, 2.2))
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.1, depth=2, location=(0, 0, 1))

    cone = bpy.data.objects['Cone']
    cylinder = bpy.data.objects['Cylinder']

    ctx = bpy.context.copy()
    ctx['active_object'] = cone
    ctx['selected_objects'] = cylinder
    ctx['selected_editable_objects'] = [cone, cylinder]
    bpy.ops.object.join(ctx)

    arrow = cone
    arrow.name = 'axis_arrow'

    arrow.select_set(True)
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    x_mat = bpy.data.materials.new("x_mat")
    x_mat.diffuse_color = (1, 0, 0, 1)
    y_mat = bpy.data.materials.new("y_mat")
    y_mat.diffuse_color = (0, 1, 0, 1)
    z_mat = bpy.data.materials.new("z_mat")
    z_mat.diffuse_color = (0, 0, 1, 1)

    me = arrow.data
    z_axis = arrow
    z_axis.name = 'z_axis'

    y_axis = bpy.data.objects.new('y_axis', me.copy())
    bpy.context.scene.collection.children[0].objects.link(y_axis)
    y_axis.rotation_euler[0] = -np.pi/2

    x_axis = bpy.data.objects.new('x_axis', me.copy())
    bpy.context.scene.collection.children[0].objects.link(x_axis)
    x_axis.rotation_euler[1] = np.pi/2

    z_axis.data.materials.append(z_mat)
    y_axis.data.materials.append(y_mat)
    x_axis.data.materials.append(x_mat)

    ctx = bpy.context.copy()
    ctx['active_object'] = z_axis
    ctx['selected_objects'] = [x_axis, y_axis, z_axis]
    ctx['selected_editable_objects'] = [x_axis, y_axis, z_axis]
    bpy.ops.object.join(ctx)

    axis = z_axis
    axis.name = name

    axis.rotation_mode = 'XYZ'
    axis.rotation_euler = rot.as_euler('xyz')
    axis.select_set(True)
    #bpy.ops.transform.translate(value=tvec, orient_type='LOCAL')
    axis.location = tvec


# rotation matrix and translate vector to transform from object to world
rot = Rotation.from_euler('XYX', [90, 90, 10], degrees=True)
tvec = np.array([1, 1, 1])
create_axis('camera_axis', rot, tvec)
