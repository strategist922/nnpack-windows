set current_dir=%cd%

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\common.obj %current_dir%\common.py

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\2d-fourier-8x8.obj %current_dir%\2d-fourier-8x8.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\2d-fourier-16x16.obj %current_dir%\2d-fourier-16x16.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\2d-winograd-8x8-3x3.obj %current_dir%\2d-winograd-8x8-3x3.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\max-pooling.obj %current_dir%\max-pooling.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\relu.obj %current_dir%\relu.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\softmax.obj %current_dir%\softmax.py

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\blas\avx.obj %current_dir%\blas\avx.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\blas\avx2.obj %current_dir%\blas\avx2.py

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\blas\s8gemm.obj %current_dir%\blas\s8gemm.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\blas\c8gemm.obj %current_dir%\blas\c8gemm.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\blas\s4c6gemm.obj %current_dir%\blas\s4c6gemm.py

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\blas\conv1x1.obj %current_dir%\blas\conv1x1.py

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\blas\sgemm.obj %current_dir%\blas\sgemm.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\blas\sdotxf.obj %current_dir%\blas\sdotxf.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\blas\shdotxf.obj %current_dir%\blas\shdotxf.py

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\fft-soa.obj %current_dir%\fft-soa.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\fft-aos.obj %current_dir%\fft-aos.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\fft-dualreal.obj %current_dir%\fft-dualreal.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\ifft-dualreal.obj %current_dir%\ifft-dualreal.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\fft-real.obj %current_dir%\fft-real.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\ifft-real.obj %current_dir%\ifft-real.py

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\winograd-f6k3.obj %current_dir%\winograd-f6k3.py


python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\fft\complex_soa.obj %current_dir%\fft\complex_soa.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\fft\complex_soa_perm_to_real.obj %current_dir%\fft\complex_soa_perm_to_real.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\fft\real_to_complex_soa_perm.obj %current_dir%\fft\real_to_complex_soa_perm.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\fft\two_complex_soa_perm_to_two_real_planar.obj %current_dir%\fft\two_complex_soa_perm_to_two_real_planar.py
python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\fft\two_real_to_two_complex_soa_perm_planar.obj %current_dir%\fft\two_real_to_two_complex_soa_perm_planar.py

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\vecmath\exp.obj %current_dir%\vecmath\exp.py

python -m peachpy.x86_64 -mabi=ms -mimage-format=ms-coff -mcpu=haswell -o %current_dir%\winograd\o6x6k3x3.obj %current_dir%\winograd\o6x6k3x3.py