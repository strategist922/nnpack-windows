SETLOCAL ENABLEEXTENSIONS

SET python_dir=C:\PROGRA~1\Python36
SET nnpack_dir=C:\Users\dhaen\Source\Repos\nnpack
SET output_dir=%nnpack_dir%\x64\Release
SET current_dir=%nnpack_dir%\src\x86_64-fma
SET proc_arch=haswell

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\common.obj %current_dir%\common.py

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\2d-fourier-8x8.obj %current_dir%\2d-fourier-8x8.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\2d-fourier-16x16.obj %current_dir%\2d-fourier-16x16.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\2d-winograd-8x8-3x3.obj %current_dir%\2d-winograd-8x8-3x3.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\max-pooling.obj %current_dir%\max-pooling.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\relu.obj %current_dir%\relu.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\softmax.obj %current_dir%\softmax.py

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\avx.obj %current_dir%\blas\avx.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\avx2.obj %current_dir%\blas\avx2.py

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\s8gemm.obj %current_dir%\blas\s8gemm.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\c8gemm.obj %current_dir%\blas\c8gemm.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\s4c6gemm.obj %current_dir%\blas\s4c6gemm.py

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\conv1x1.obj %current_dir%\blas\conv1x1.py

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\sgemm.obj %current_dir%\blas\sgemm.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\sdotxf.obj %current_dir%\blas\sdotxf.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\shdotxf.obj %current_dir%\blas\shdotxf.py

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\fft-soa.obj %current_dir%\fft-soa.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\fft-aos.obj %current_dir%\fft-aos.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\fft-dualreal.obj %current_dir%\fft-dualreal.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\ifft-dualreal.obj %current_dir%\ifft-dualreal.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\fft-real.obj %current_dir%\fft-real.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\ifft-real.obj %current_dir%\ifft-real.py

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\winograd-f6k3.obj %current_dir%\winograd-f6k3.py

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\complex_soa.obj %current_dir%\fft\complex_soa.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\complex_soa_perm_to_real.obj %current_dir%\fft\complex_soa_perm_to_real.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\real_to_complex_soa_perm.obj %current_dir%\fft\real_to_complex_soa_perm.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\two_complex_soa_perm_to_two_real_planar.obj %current_dir%\fft\two_complex_soa_perm_to_two_real_planar.py
%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\two_real_to_two_complex_soa_perm_planar.obj %current_dir%\fft\two_real_to_two_complex_soa_perm_planar.py

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\exp.obj %current_dir%\vecmath\exp.py

%python_dir%\python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\o6x6k3x3.obj %current_dir%\winograd\o6x6k3x3.py

ENDLOCAL

