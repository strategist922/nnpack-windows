setlocal enableextensions

set nnpack_dir=%~1

set source_dir=%nnpack_dir%src\x86_64-fma
set output_dir=%nnpack_dir%x64\Debug
set python_dir=C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python36_64\
set proc_arch=haswell

cd "%source_dir%"

"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\2d-fourier-8x8.obj "%source_dir%"\2d-fourier-8x8.py
"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\2d-fourier-16x16.obj "%source_dir%"\2d-fourier-16x16.py
"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\2d-winograd-8x8-3x3.obj "%source_dir%"\2d-winograd-8x8-3x3.py

"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\s8gemm.obj "%source_dir%"\blas\s8gemm.py
"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\c8gemm.obj "%source_dir%"\blas\c8gemm.py
"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\s4c6gemm.obj "%source_dir%"\blas\s4c6gemm.py

"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\conv1x1.obj "%source_dir%"\blas\conv1x1.py

"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\sgemm.obj "%source_dir%"\blas\sgemm.py

"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\max-pooling.obj "%source_dir%"\max-pooling.py
"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\relu.obj "%source_dir%"\relu.py
"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\softmaxpy.obj "%source_dir%"\softmax.py

"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\sdotxf.obj "%source_dir%"\blas\sdotxf.py
"%python_dir%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o "%output_dir%"\shdotxf.obj "%source_dir%"\blas\shdotxf.py

endlocal
exit
