# OpenFOAM Mesh Importer

  A Blender addon for OpenFOAM mesh importing. Mesh data and boundary conditions can be imported to Blender for further visualization analysis.

  Tested in Blender version 3.4.1, maybe work in Blender>=2.80.
  Tested OpenFOAM 10 example case: `$FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity`

## install
  Download file `import_OpenFOAM_mesh.py`.
  Open Blender, Click `Edit` -> `Preferences` -> `Add-ons` -> `Install...`.
  Select the `io_import_OpenFOAM_mesh.py` downloaded previously.
  Check the box of `OpenFOAM Mesh Importer` addon.

## Usage
  Open Blender, Click `File` -> `Import` -> `Import OpenFOAM Mesh`.
  Select the polyMesh directory (eg. "constant/polyMesh" under case directory) and import.

## File Statement
  io_import_OpenFOAM_mesh.py: main core of the mesh importer addon.
  cavity: test example from OpenFOAM 10 tutorial. select this folder when testing the addon.

## Dependency
  [openfoamparser](https://github.com/ApolloLV/openfoamparser): Corresponding parts have already been integrated in `io_import_OpenFOAM_mesh.py`.
  numpy: already bundled with Blender>=2.80.
  Therefore, no additional installation needed for dependency requirements.

## License
  MIT

## Author
  OpenFOAM Mesh Importer: SUN Smallwhite <niasw@pku.edu.cn>
  [openfoamparser](https://github.com/ApolloLV/openfoamparser): XU Xianghua, Jan Drees, Timothy-Edward-Kendon, YuyangL

