# NNPACK for Windows 
#### port of Marat Dukhan NNPACK - BSD 2-Clause "Simplified" (https://github.com/Maratyszcza/NNPACK)


Before building this repo you have to install PeachPy.

Open a Phyton command prompt with Administrator rights and type:
  
  pip install --upgrade git+https://github.com/Maratyszcza/PeachPy

Now you're ready for building with Visual Studio 2017


Results of the unit tests:

### convolution-output:

  * FT8x8:    passed

  * FT16x16:  failed

  * WT8x8:    passed


### convolution-input-gradient:

  * FT8x8:    passed except FT8x8.few_output_channels

  * FT16x16:  failed

  * WT8x8:    passed except WT8x8.few_output_channels


### convolution-kernel-gradient:

  * FT8x8:    passed

  * FT16x16:  failed

  * WT8x8:    disabled


### convolution-inference:

  * implicit gemm:  passed
  
  * direct convolution: passed
  
  * FT8x8:          passed
  
  * FT16x16:        failed
  
  * WT8x8:          passed
  

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
