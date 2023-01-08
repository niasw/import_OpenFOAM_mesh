bl_info = {
    'name': 'OpenFOAM Mesh Importer',
    'category': 'Import-Export',
    'author': 'Sun Smallwhite <niasw@pku.edu.cn>',
    'location': 'File > Import > Import OpenFOAM Mesh',
    'description': 'Import OpenFOAM Mesh',
    'version': (1, 0, 0),
    'blender': (2, 80, 0),
    'wiki_url': 'https://github.com/niasw/import_openfoam_mesh',
    'tracker_url': 'https://github.com/niasw/import_openfoam_mesh/issues',
    'license': 'MIT',
    'warning': '',
}

'''
MIT License

OpenFOAM Mesh Importer:
Copyright (c) 2022 Sun Smallwhite

openfoamparser:
Copyright (c) 2017 dayigu, 2019 Jan Drees, Timothy-Edward-Kendon, YuyangL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import bpy
import csv
from bpy_extras import object_utils
from bpy.props import (
    StringProperty,
    EnumProperty
)
from mathutils import (
    Vector
)
from bpy_extras.io_utils import (
    ImportHelper
)

# ------------------------- Begin of quoting openfoamparser.mesh_parser -------------------------

import numpy as np
import os
import re
import struct
from collections import namedtuple

# ------------------------- End of quoting openfoamparser.mesh_parser -------------------------

class ImportOpenFOAMmesh(bpy.types.Operator, ImportHelper):
    """Import OpenFOAM mesh"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "import_openfoam.import_mesh"        # Unique identifier for buttons and menu items to reference.
    bl_label = "Import OpenFOAM Mesh"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    filepath: StringProperty(
        name="input polyMesh directory",
        subtype="DIR_PATH",
    )

    cellmode: EnumProperty(
        name="cell mode",
        items=(
            ('off', "off", "whole mesh as an object."),
            ('cell', "cell", "one cell one object."),
        ),
    )

    objname: StringProperty(
        name="input polyMesh directory",
        default="OpenFOAM_mesh",
        subtype="NONE",
    )

    def execute(self, context):        # execute() is called when running the operator.

        pathname =  self.filepath
        
        fm = FoamMesh(self.filepath)
        self.report({'INFO'}, 'OpenFOAM mesh Imported.')


        if (bpy.context.mode=="EDIT_MESH"):
            bpy.ops.object.mode_set(mode='OBJECT')

        # handle with materials by boundary info
        mtr=bpy.data.materials.new('0.InnerFace')
        mtr.diffuse_color=(0.8,0.8,0.8,0.6) # appearance color
        mtrs=[mtr]
        
        # make id->name mapping of boundary info
        bd_names={};
        num_bd=len(fm.boundary)
        for bd_name in fm.boundary:
            bd_names[fm.boundary[bd_name][3]]=bd_name
        for bd_id in bd_names:
            mtr=bpy.data.materials.new(str(abs(bd_id))+'.'+bd_names[bd_id].decode())
            mtr.diffuse_color=wheelColor(-float(bd_id)/num_bd)
            mtrs.append(mtr)

        if (self.cellmode=='off'):
            (verts, edges, faces)=formatVertsEdgesFaces(fm.points, fm.faces)
            obj=createMeshObj(context, verts, edges, faces, self.objname)
            self.report({'INFO'}, '1 mesh Created.')
            bpy.context.view_layer.objects.active=obj
            for imtr in range(len(mtrs)):
                bpy.ops.object.material_slot_add()
                obj.data.materials[imtr]=mtrs[imtr]
            for bd_id in bd_names:
                bd=fm.boundary[bd_names[bd_id]]
                for ifc in range(bd[2],bd[2]+bd[1]):
                    obj.data.polygons[ifc].material_index=abs(bd_id)
        elif (self.cellmode=='cell'):
            #fm.cells=cellReader(fm) #deprecated
            for cit in range(fm.num_cell):
                #(verts, edges, faces)=formatVertsEdgesFaces_cell(fm.points, fm.faces, fm.cells[cit]) #deprecated cellReader's data struct
                (verts, edges, faces)=formatVertsEdgesFaces_cell(fm.points, fm.faces, fm.cell_faces[cit])
                obj=createMeshObj(context, verts, edges, faces, self.objname+"_"+str(cit))
                bpy.context.view_layer.objects.active=obj
                for imtr in range(len(mtrs)):
                    bpy.ops.object.material_slot_add()
                    obj.data.materials[imtr]=mtrs[imtr]
                for ifc in range(len(fm.cell_faces[cit])):
                    ibd=fm.cell_neighbour_cells(cit)[ifc]
                    if (ibd<0):
                        obj.data.polygons[ifc].material_index=abs(ibd)
            self.report({'INFO'}, str(fm.num_cell)+' mesh Created.')
        else:
            self.report({'ERROR'}, 'Unknown cell mode: '+self.cellmode+'.')
        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

# ------------------------- Begin of quoting openfoamparser.mesh_parser -------------------------

OpenFOAM_Boundary = namedtuple('OpenFOAM_Boundary', 'type, num, start, id')

def is_integer(s):
    try:
        x = int(s)
        return True
    except ValueError:
        return False

def is_binary_format(content, maxline=20):
    """
    parse file header to judge the format is binary or not
    :param content: file content in line list
    :param maxline: maximum lines to parse
    :return: binary format or not
    """
    for lc in content[:maxline]:
        if b'format' in lc:
            if b'binary' in lc:
                return True
            return False
    return False

def parse_data_uniform(line):
    """
    parse uniform data from a line
    :param line: a line include uniform data, eg. "value           uniform (0 0 0);"
    :return: data
    """
    if b'(' in line:
        return np.array([float(x) for x in line.split(b'(')[1].split(b')')[0].split()])
    return float(line.split(b'uniform')[1].split(b';')[0])


def parse_data_nonuniform(content, n, n2, is_binary):
    """
    parse nonuniform data from lines
    :param content: data content
    :param n: line number
    :param n2: last line number
    :param is_binary: binary format or not
    :return: data
    """
    num = int(content[n + 1])
    if not is_binary:
        if b'scalar' in content[n]:
            data = np.array([float(x) for x in content[n + 3:n + 3 + num]])
        else:
            data = np.array([ln[1:-2].split() for ln in content[n + 3:n + 3 + num]], dtype=float)
    else:
        nn = 1
        if b'vector' in content[n]:
            nn = 3
        elif b'symmTensor' in content[n]:
            nn = 6
        elif b'tensor' in content[n]:
            nn = 9
        buf = b''.join(content[n+2:n2+1])
        vv = np.array(struct.unpack('{}d'.format(num*nn),
                                    buf[struct.calcsize('c'):num*nn*struct.calcsize('d')+struct.calcsize('c')]))
        if nn > 1:
            data = vv.reshape((num, nn))
        else:
            data = vv
    return data

def parse_internal_field_content(content):
    """
    parse internal field from content
    :param content: contents of lines
    :return: numpy array of internal field
    """
    is_binary = is_binary_format(content)
    for ln, lc in enumerate(content):
        if lc.startswith(b'internalField'):
            if b'nonuniform' in lc:
                return parse_data_nonuniform(content, ln, len(content), is_binary)
            elif b'uniform' in lc:
                return parse_data_uniform(content[ln])
            break
    return None

def parse_internal_field(fn):
    """
    parse internal field, extract data to numpy.array
    :param fn: file name
    :return: numpy array of internal field
    """
    if not os.path.exists(fn):
        print("Can not open file " + fn)
        return None
    with open(fn, "rb") as f:
        content = f.readlines()
        return parse_internal_field_content(content)


class FoamMesh(object):
    """ FoamMesh class """
    def __init__(self, path):
        self.path = path
        self.__cell_neighbour_constructed=False
        self._parse_mesh_data(self.path)
        self.num_point = len(self.points)
        self.num_face = len(self.owner)
        self.num_inner_face = len(self.neighbour)
        self.num_cell = max(max(self.owner), max(self.neighbour)) + 1
        self._set_boundary_faces() # append boundary faces data into 'neighbour' variable, boundary faces are indexed by negative integers
        self._construct_cells() # cell data => self.cell_faces, neighbour cells and boundaries => self.cell_neighbour
        self.cell_centres = None
        self.cell_volumes = None
        self.face_areas = None

    def read_cell_centres(self, fn):
        """
        read cell centres coordinates from data file,
        the file can be got by `postProcess -func 'writeCellCentres' -time 0'
        :param fn: cell centres file name, eg. '0/C'
        :return: None
        """
        self.cell_centres = parse_internal_field(fn)

    def read_cell_volumes(self, fn):
        """
        read cell volumes from data file,
        the file can be got by `postProcess -func 'writeCellVolumes' -time 0'
        :param fn: cell volumes file name, eg. '0/V'
        :return: None
        """
        self.cell_volumes = parse_internal_field(fn)

#    def read_face_areas(self, fn):
#        """
#        read face areas from data file,
#        """
#        self.face_areas = parse_internal_field(fn)

    def cell_neighbour_cells(self, i):
        """
        return neighbour cells of cell i
        :param i: cell index
        :return: neighbour cell list
        """
        if self.__cell_neighbour_constructed is not True:
            self._construct_cell_neighbour()
        return self.cell_neighbour[i]

    def is_cell_on_boundary(self, i, bd=None):
        """
        check if cell i is on boundary bd
        :param i: cell index, 0<=i<num_cell
        :param bd: boundary name, byte str
        :return: True or False
        """
        if self.__cell_neighbour_constructed is not True:
            self._construct_cell_neighbour()
        if i < 0 or i >= self.num_cell:
            return False
        if bd is not None:
            try:
                bid = self.boundary[bd].id
            except KeyError:
                return False
        for n in self.cell_neighbour[i]:
            if bd is None and n < 0:
                return True
            elif bd and n == bid:
                return True
        return False

    def is_face_on_boundary(self, i, bd=None):
        """
        check if face i is on boundary bd
        :param i: face index, 0<=i<num_face
        :param bd: boundary name, byte str
        :return: True or False
        """
        if i < 0 or i >= self.num_face:
            return False
        if bd is None:
            if self.neighbour[i] < 0:
                return True
            return False
        try:
            bid = self.boundary[bd].id
        except KeyError:
            return False
        if self.neighbour[i] == bid:
            return True
        return False

    def boundary_cells(self, bd):
        """
        return cell id list on boundary bd
        :param bd: boundary name, byte str
        :return: cell id generator
        """
        try:
            b = self.boundary[bd]
            return (self.owner[f] for f in range(b.start, b.start+b.num))
        except KeyError:
            return ()

    def _set_boundary_faces(self):
        """
        set faces' boundary id which on boundary
        :return: none
        """
        self.neighbour.extend([-1]*(self.num_face - self.num_inner_face))
        for b in self.boundary.values():
            self.neighbour[b.start:b.start+b.num] = [b.id]*b.num

    def _construct_cells(self): # modified to make mapping order consistent with cell_neighbour
        """
        construct cell faces
        :return: none
        """
        self.cell_faces = [[] for i in range(self.num_cell)]
        for ifc, icl in enumerate(self.owner):
            self.cell_faces[icl].append(ifc)
        for ifc in range(self.num_inner_face):
            self.cell_faces[self.neighbour[ifc]].append(ifc)

    def _construct_cell_neighbour(self): # modified to make mapping order consistent with cell_neighbour
        """
        construct cell neighbours
        :return: none
        """
        if self.__cell_neighbour_constructed is not True:
            self.cell_neighbour = [[] for i in range(self.num_cell)]
            for icl in range(self.num_cell):
                for ifc in self.cell_faces[icl]:
                    if (icl==self.owner[ifc]): # owner face
                        self.cell_neighbour[icl].append(self.neighbour[ifc])
                    else: # neighbour face
                        self.cell_neighbour[icl].append(self.owner[ifc])
            self.__cell_neighbour_constructed=True

    def _parse_mesh_data(self, path):
        """
        parse mesh data from mesh files
        :param path: path of mesh files
        :return: none
        """
        self.boundary = self.parse_mesh_file(os.path.join(path, 'boundary'), self.parse_boundary_content)
        self.points = self.parse_mesh_file(os.path.join(path, 'points'), self.parse_points_content)
        self.faces = self.parse_mesh_file(os.path.join(path, 'faces'), self.parse_faces_content)
        self.owner = self.parse_mesh_file(os.path.join(path, 'owner'), self.parse_owner_neighbour_content)
        self.neighbour = self.parse_mesh_file(os.path.join(path, 'neighbour'), self.parse_owner_neighbour_content)

    @classmethod
    def parse_mesh_file(cls, fn, parser):
        """
        parse mesh file
        :param fn: boundary file name
        :param parser: parser of the mesh
        :return: mesh data
        """
        try:
            with open(fn, "rb") as f:
                content = f.readlines()
                return parser(content, is_binary_format(content))
        except FileNotFoundError:
            print('file not found: %s'%fn)
            return None

    @classmethod
    def parse_points_content(cls, content, is_binary, skip=10):
        """
        parse points from content
        :param content: file contents
        :param is_binary: binary format or not
        :param skip: skip lines
        :return: points coordinates as numpy.array
        """
        n = skip
        while n < len(content):
            lc = content[n]
            if is_integer(lc):
                num = int(lc)
                if not is_binary:
                    data = np.array([ln[1:-2].split() for ln in content[n + 2:n + 2 + num]], dtype=float)
                else:
                    buf = b''.join(content[n+1:])
                    disp = struct.calcsize('c')
                    vv = np.array(struct.unpack('{}d'.format(num*3),
                                                buf[disp:num*3*struct.calcsize('d') + disp]))
                    data = vv.reshape((num, 3))
                return data
            n += 1
        return None


    @classmethod
    def parse_owner_neighbour_content(cls, content, is_binary, skip=10):
        """
        parse owner or neighbour from content
        :param content: file contents
        :param is_binary: binary format or not
        :param skip: skip lines
        :return: indexes as list
        """
        n = skip
        while n < len(content):
            lc = content[n]
            if is_integer(lc):
                num = int(lc)
                if not is_binary:
                    data = [int(ln) for ln in content[n + 2:n + 2 + num]]
                else:
                    buf = b''.join(content[n+1:])
                    disp = struct.calcsize('c')
                    data = struct.unpack('{}i'.format(num),
                                         buf[disp:num*struct.calcsize('i') + disp])
                return list(data)
            n += 1
        return None

    @classmethod
    def parse_faces_content(cls, content, is_binary, skip=10):
        """
        parse faces from content
        :param content: file contents
        :param is_binary: binary format or not
        :param skip: skip lines
        :return: faces as list
        """
        n = skip
        while n < len(content):
            lc = content[n]
            if is_integer(lc):
                num = int(lc)
                if not is_binary:
                    data = [[int(s) for s in re.findall(b"\d+", ln)[1:]] for ln in content[n + 2:n + 2 + num]]
                else:
                    buf = b''.join(content[n+1:])
                    disp = struct.calcsize('c')
                    idx = struct.unpack('{}i'.format(num), buf[disp:num*struct.calcsize('i') + disp])
                    disp = 3*struct.calcsize('c') + 2*struct.calcsize('i')
                    pp = struct.unpack('{}i'.format(idx[-1]),
                                       buf[disp+num*struct.calcsize('i'):
                                           disp+(num+idx[-1])*struct.calcsize('i')])
                    data = []
                    for i in range(num - 1):
                        data.append(pp[idx[i]:idx[i+1]])
                return data
            n += 1
        return None

    @classmethod
    def parse_boundary_content(cls, content, is_binary=None, skip=10):
        """
        parse boundary from content
        :param content: file contents
        :param is_binary: binary format or not, not used
        :param skip: skip lines
        :return: boundary dict
        """
        bd = {}
        num_boundary = 0
        n = skip
        bid = 0
        in_boundary_field = False
        in_patch_field = False
        current_patch = b''
        current_type = b''
        current_nFaces = 0
        current_start = 0
        while True:
            if n > len(content):
                if in_boundary_field:
                    print('error, boundaryField not end with )')
                break
            lc = content[n]
            if not in_boundary_field:
                if is_integer(lc.strip()):
                    num_boundary = int(lc.strip())
                    in_boundary_field = True
                    if content[n + 1].startswith(b'('):
                        n += 2
                        continue
                    elif content[n + 1].strip() == b'' and content[n + 2].startswith(b'('):
                        n += 3
                        continue
                    else:
                        print('no ( after boundary number')
                        break
            if in_boundary_field:
                if lc.startswith(b')'):
                    break
                if in_patch_field:
                    if lc.strip() == b'}':
                        in_patch_field = False
                        bd[current_patch] = OpenFOAM_Boundary(current_type, current_nFaces, current_start, -1-bid)
                        bid += 1
                        current_patch = b''
                    elif b'nFaces' in lc:
                        current_nFaces = int(lc.split()[1][:-1])
                    elif b'startFace' in lc:
                        current_start = int(lc.split()[1][:-1])
                    elif b'type' in lc:
                        current_type = lc.split()[1][:-1]
                else:
                    if lc.strip() == b'':
                        n += 1
                        continue
                    current_patch = lc.strip()
                    if content[n + 1].strip() == b'{':
                        n += 2
                    elif content[n + 1].strip() == b'' and content[n + 2].strip() == b'{':
                        n += 3
                    else:
                        print('no { after boundary patch')
                        break
                    in_patch_field = True
                    continue
            n += 1

        return bd

# ------------------------- End of quoting openfoamparser.mesh_parser -------------------------

def wheelColor(x):
    '''
    Mapping 0-1 to Wheel color
    '''
    if (x<=0):
        return (0.8,0.2,0.2,0.6)
    elif (x<=0.33):
        return (0.8-0.6*x/0.33,0.2+0.6*x/0.33,0.2,0.6)
    elif (x<=0.67):
        return (0.2,0.8-0.6*(x-0.33)/0.34,0.2+0.6*(x-0.33)/0.34,0.6)
    elif (x<=1):
        return (0.2+0.6*(x-0.67)/0.33,0.2,0.8-0.6*(x-0.67)/0.33,0.6)
    else:
        return (0.8,0.2,0.2,0.6)

def createMeshObj(context, verts, edges, faces, meshname):
    mesh=bpy.data.meshes.new(meshname)
    mesh.from_pydata(verts,edges,faces)
    mesh.use_auto_smooth=False
    mesh.update()

    if (bpy.context.mode=="EDIT_MESH"):
        bpy.ops.object.mode_set(mode='OBJECT')

    obj=object_utils.object_data_add(context,mesh)

    return obj

def formatVertsEdgesFaces(points, faces):
    '''
    Transfer OpenFOAM mesh (points, faces) into Blender mesh (verts, edges, faces)
    '''
    verts=[Vector(it) for it in points]
    edges=[]
    record={}
    for fc in faces:
        for pit in range(len(fc)-1):
            lined=False
            if (fc[pit] in record):
                if (fc[pit+1] in record[fc[pit]]):
                    lined=True
            if (fc[pit+1] in record):
                if (fc[pit] in record[fc[pit+1]]):
                    lined=True
            if (not lined):
                if fc[pit] in record:
                    record[fc[pit]].append(fc[pit+1])
                else:
                    record[fc[pit]]=[fc[pit+1]]
                edges.append([fc[pit],fc[pit+1]])
        lined=False
        if (fc[len(fc)-1] in record):
            if (fc[0] in record[fc[len(fc)-1]]):
                lined=True
        if (fc[0] in record):
            if (fc[len(fc)-1] in record[fc[0]]):
                lined=True
        if (not lined):
            if fc[len(fc)-1] in record:
                record[fc[len(fc)-1]].append(fc[0])
            else:
                record[fc[len(fc)-1]]=[fc[0]]
            edges.append([fc[len(fc)-1],fc[0]])
    return (verts, edges, faces);

#def cellReader(fm):
#    '''
#    (backup realization of FoamMesh._construct_cells)
#    (advantage for face analysis, disadvantage for cell analysis)
#    read out cells from OpenFOAM mesh
#    faceID:
#        if owner: faceID+1
#        if neighbour: -faceID-1
#        avoid 0 for confusion.
#    '''
#    cells=[]
#    cell_dict={}
#    for ifc in range(len(fm.owner)):
#        if fm.owner[ifc] in cell_dict:
#            cell_dict[fm.owner[ifc]].append(ifc+1)
#        else:
#            cell_dict[fm.owner[ifc]]=[ifc+1]
#    for ifc in range(len(fm.neighbour)):
#        if fm.neighbour[ifc] in cell_dict:
#            cell_dict[fm.neighbour[ifc]].append(-ifc-1)
#        else:
#            cell_dict[fm.neighbour[ifc]]=[-ifc-1]
#    for itc in cell_dict:
#        cells.append(cell_dict[itc])
#    return cells

#def formatVertsEdgesFaces_cell(points, faces, cell):
#    '''
#    (backup realization. using deprecated function cellReader's cell data struct)
#    Transfer one cell in OpenFOAM mesh (points, faces) into Blender mesh (verts, edges, faces)
#    '''
#    verts=[]
#    vert_dict={}
#    edges=[]
#    record={}
#    mapvts={}
#    for ifc_drct in cell:
#        ifc=abs(ifc_drct)-1
#        fc=faces[ifc]
#        for pit in range(len(fc)-1):
#            vert_dict[fc[pit]]=True
#            lined=False
#            if (fc[pit] in record):
#                if (fc[pit+1] in record[fc[pit]]):
#                    lined=True
#            if (fc[pit+1] in record):
#                if (fc[pit] in record[fc[pit+1]]):
#                    lined=True
#            if (not lined):
#                if fc[pit] in record:
#                    record[fc[pit]].append(fc[pit+1])
#                else:
#                    record[fc[pit]]=[fc[pit+1]]
#                edges.append([fc[pit],fc[pit+1]])
#        lined=False
#        if (fc[len(fc)-1] in record):
#            if (fc[0] in record[fc[len(fc)-1]]):
#                lined=True
#        if (fc[0] in record):
#            if (fc[len(fc)-1] in record[fc[0]]):
#                lined=True
#        if (not lined):
#            if fc[len(fc)-1] in record:
#                record[fc[len(fc)-1]].append(fc[0])
#            else:
#                record[fc[len(fc)-1]]=[fc[0]]
#            edges.append([fc[len(fc)-1],fc[0]])
#    for itv in vert_dict:
#        mapvts[itv]=len(verts)
#        verts.append(Vector(points[itv]))
#    cellfaces=[]
#    celledges=[[mapvts[edg[0]],mapvts[edg[1]]] for edg in edges]
#    for ifc_drct in cell:
#        ifc=abs(ifc_drct)-1
#        fc=faces[ifc]
#        cellfaces.append([mapvts[itv] for itv in fc])
#    return (verts, celledges, cellfaces);

def formatVertsEdgesFaces_cell(points, faces, cell):
    '''
    Transfer one cell in OpenFOAM mesh (points, faces) into Blender mesh (verts, edges, faces)
    '''
    verts=[]
    vert_dict={}
    edges=[]
    record={}
    mapvts={}
    for ifc in cell:
        fc=faces[ifc]
        for pit in range(len(fc)-1):
            vert_dict[fc[pit]]=True
            lined=False
            if (fc[pit] in record):
                if (fc[pit+1] in record[fc[pit]]):
                    lined=True
            if (fc[pit+1] in record):
                if (fc[pit] in record[fc[pit+1]]):
                    lined=True
            if (not lined):
                if fc[pit] in record:
                    record[fc[pit]].append(fc[pit+1])
                else:
                    record[fc[pit]]=[fc[pit+1]]
                edges.append([fc[pit],fc[pit+1]])
        lined=False
        if (fc[len(fc)-1] in record):
            if (fc[0] in record[fc[len(fc)-1]]):
                lined=True
        if (fc[0] in record):
            if (fc[len(fc)-1] in record[fc[0]]):
                lined=True
        if (not lined):
            if fc[len(fc)-1] in record:
                record[fc[len(fc)-1]].append(fc[0])
            else:
                record[fc[len(fc)-1]]=[fc[0]]
            edges.append([fc[len(fc)-1],fc[0]])
    for itv in vert_dict:
        mapvts[itv]=len(verts)
        verts.append(Vector(points[itv]))
    cellfaces=[]
    celledges=[[mapvts[edg[0]],mapvts[edg[1]]] for edg in edges]
    for ifc in cell:
        fc=faces[ifc]
        cellfaces.append([mapvts[itv] for itv in fc])
    return (verts, celledges, cellfaces);

def menu_func(self, context):
    self.layout.operator(ImportOpenFOAMmesh.bl_idname, text=ImportOpenFOAMmesh.bl_label)

def register():
    bpy.utils.register_class(ImportOpenFOAMmesh)
    bpy.types.TOPBAR_MT_file_import.append(menu_func)  # Adds the new operator to an existing menu.

def unregister():
    bpy.utils.unregister_class(ImportOpenFOAMmesh)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func)  # Removes the operator from the existing menu.

# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()
