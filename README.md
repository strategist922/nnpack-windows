# nnpack-windows
nnpack for Windows

Windows port of Marat Dukhan NNPACK - BSD 2-Clause "Simplified" (https://github.com/Maratyszcza/NNPACK)

the included objects are created like this:

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o avx.obj avx.py

if you have a different cpu you need to change -mcp=skylake for example
