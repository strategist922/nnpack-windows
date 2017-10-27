setlocal enableextensions

set nnpack_dir=%~1

set source_dir=%nnpack_dir%src\x86_64-fma
set output_dir=%nnpack_dir%x64\Debug
set proc_arch=haswell

cd %source_dir%

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\common.obj %source_dir%\common.py

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\2d-fourier-8x8.obj %source_dir%\2d-fourier-8x8.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\2d-fourier-16x16.obj %source_dir%\2d-fourier-16x16.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\2d-winograd-8x8-3x3.obj %source_dir%\2d-winograd-8x8-3x3.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\max-pooling.obj %source_dir%\max-pooling.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\relu.obj %source_dir%\relu.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\softmaxpy.obj %source_dir%\softmax.py

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\avx.obj %source_dir%\blas\avx.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\avx2.obj %source_dir%\blas\avx2.py

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\s8gemm.obj %source_dir%\blas\s8gemm.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\c8gemm.obj %source_dir%\blas\c8gemm.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\s4c6gemm.obj %source_dir%\blas\s4c6gemm.py

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\conv1x1.obj %source_dir%\blas\conv1x1.py

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\sgemm.obj %source_dir%\blas\sgemm.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\sdotxf.obj %source_dir%\blas\sdotxf.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\shdotxf.obj %source_dir%\blas\shdotxf.py

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\fft-soa.obj %source_dir%\fft-soa.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\fft-aos.obj %source_dir%\fft-aos.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\fft-dualreal.obj %source_dir%\fft-dualreal.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\ifft-dualreal.obj %source_dir%\ifft-dualreal.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\fft-real.obj %source_dir%\fft-real.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\ifft-real.obj %source_dir%\ifft-real.py

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\winograd-f6k3.obj %source_dir%\winograd-f6k3.py

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\complex_soa.obj %source_dir%\fft\complex_soa.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\complex_soa_perm_to_real.obj %source_dir%\fft\complex_soa_perm_to_real.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\real_to_complex_soa_perm.obj %source_dir%\fft\real_to_complex_soa_perm.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\two_complex_soa_perm_to_two_real_planar.obj %source_dir%\fft\two_complex_soa_perm_to_two_real_planar.py
"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\two_real_to_two_complex_soa_perm_planar.obj %source_dir%\fft\two_real_to_two_complex_soa_perm_planar.py

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\exp.obj %source_dir%\vecmath\exp.py

"%PYTHONPATH%"python.exe -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=%proc_arch% -o %output_dir%\o6x6k3x3.obj %source_dir%\winograd\o6x6k3x3.py

endlocal
exit

