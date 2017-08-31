# nnpack-windows
nnpack for Windows

Windows port of Marat Dukhan NNPACK - BSD 2-Clause "Simplified" (https://github.com/Maratyszcza/NNPACK)

The steps to build the nnpack-windows repo:

(Check you have Visual Studio 2017 and Python installed)

To install PeachPy:
Open a Phyton command prompt:
pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
Download the nnpack-windows repo somewhere and after extracting it 
go in the python prompt to the directory ..\nnpack-windows\src\x86_64-fma

Type: python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o common.obj common.py

and do the same for every python script in each directory. In the "blas" directory you must first create the avx.obj and avx2.obj object including those in the "fp16" subdirectory.
after all the objects are created you open te visual studio solution file with vs2017

if you have a different cpu you may need to change -mcp=skylake for example
