# Note to self : How to use bpy

Blender 2.80

## Settings

### Settings for 3D printing

```py
bpy.context.scene.unit_settings.length_unit = 'METERS'
bpy.context.scene.unit_settings.scale_length = 0.001  # mm
bpy.context.space_data.overlay.grid_scale = 0.01 # grid per 1cm
```

## Mesh

### Create mesh object from vertices

1. Create empty mesh and set vertices and faces.
2. Create mesh object.
3. Link the object to the collection.

```py
WIDTH = 75
HEIGHT = 30
THICK = 10

verts = [
    [0, 0, 0], [WIDTH, 0, 0], [WIDTH, THICK, 0], [0, THICK, 0],
    [0, 0, HEIGHT], [WIDTH, 0, HEIGHT],
    [WIDTH, THICK, HEIGHT], [0, THICK, HEIGHT],
]
faces = [[0, 1, 2, 3], [0, 4, 5, 1], [1, 5, 6, 2],
         [2, 6, 7, 3], [0, 3, 7, 4], [4, 7, 6, 5]]

msh = bpy.data.meshes.new(name="CubeMesh")
msh.from_pydata(verts, [], faces)  # vertices, edges, faces
msh.update()

obj = bpy.data.objects.new(name="Cube", object_data=msh)

bpy.context.scene.collection.objects.link(obj)
# if you want to add it to the same layer as the default camera and light.
# bpy.context.scene.collection.children[0].objects.link(obj)
```

### Delete mesh object completely

1. Unlink from the collection.
2. Remove the object data.
3. Remove the mesh data.

Note : `Delete` command in the GUI executes only step 1 and 2.

```py
bpy.context.scene.collection.objects.unlink(bpy.context.scene.collection.objects["Cube"])

bpy.data.objects.remove(bpy.data.objects["Cube"])

bpy.data.meshes.remove(bpy.data.meshes["CubeMesh"])
```

## Move object

```py
obj.location = [10.0, 0.0, 0.0]
obj.rotation_euler = [0.0, 1.57, 0.0]
```

## Modifier

### Boolean modifier

```py
# get objects
obj1 = bpy.data.objects['obj1']
obj2 = bpy.data.objects['obj2']

bool_mod = obj1.modifiers.new(type="BOOLEAN", name="MyBoolMod")
bool_mod.object = obj2
bool_mod.operation = "DIFFERENCE"
bpy.ops.object.modifier_apply(modifier=bool_mod.name)
```

