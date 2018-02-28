[![BSD (2 clause) License](https://img.shields.io/badge/License-BSD%202--Clause%20%22Simplified%22%20License-blue.svg)](https://github.com/Maratyszcza/NNPACK/blob/master/LICENSE)
# NNPACK for Windows (AVX2 backend)
[![HitCount](http://hits.dwyl.io/zeno40/nnpack-windows.svg)](http://hits.dwyl.io/zeno40/nnpack-windows)
#### port of Marat Dukhan NNPACK (https://github.com/Maratyszcza/NNPACK)

Before building this repo you have to install PeachPy.

Open a Phyton command prompt with Administrator rights and type:
  ```bash
pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
```
  
Now you're ready for building with Visual Studio 2017


Results of the unit tests:

### convolution-inference:

  * implicit gemm:  passed
  
  * direct conv:	passed
  
  * FT8x8:          passed
  
  * FT16x16:        failed


  * WT8x8:          passed
  
### convolution-output:

  * FT8x8:    passed

  * FT16x16:  failed

  * WT8x8:    passed


### convolution-input-gradient:

  * FT8x8:    passed

  * FT16x16:  failed

  * WT8x8:    passed


### convolution-kernel-gradient:

  * FT8x8:    passed

  * FT16x16:  failed

  * WT8x8:    disabled

 
### fourier:
### fully-connected-inference:
### fully-connected:
### max-pooling-output:
### relu-input-gradient:
### relu-output
### sgemm:
### softmax-output:
### winograd:

  * all passed
  
This ported version of NNPACK runs and can be compiled without modification on Linux (probably OS X/Android as well, I haven't verified this). Under these operating systems all kernels (including FT16x16) are passing the unit tests. 

```bash
git clone https://github.com/zeno40/nnpack-windows.git
cd nnpack-windows
confu setup
python ./configure.py
ninja
ninja smoketest
```
