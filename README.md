

# nnpack-windows
NNPACK for Windows

Windows port of Marat Dukhan NNPACK - BSD 2-Clause "Simplified" (https://github.com/Maratyszcza/NNPACK)


The steps to build the nnpack-windows repo:


Install PeachPy:
Open a Phyton command prompt with Administrator rights and type:
  
  pip install --upgrade git+https://github.com/Maratyszcza/PeachPy


Now you can build the repo in VS2017



Results of the unit tests:

convolution-output:

FT8x8   SUCCESSFULL

FT16x16 FULL FAIL

WT8x8	  SUCCESSFULL


convolution-input-gradient:

FT8x8   SUCCESSFULL except FT8x8.few_output_channels

FT16x16	FULL FAIL

WT8x8	  SUCCESSFULL except WT8x8.few_output_channels


convolution-kernel-gradient:

FT8x8   SUCCESSFULL

FT16x16	FULL FAIL

WT8x8:	DISABLED


convolution-inference:

FAILED


fourier:

FULL SUCCESS


fully-connected-inference:

SUCCESSFULL


fully-coneected:

SUCCESSFULL


max-pooling-output:

SUCCESSFULL


relu-input-gradient:

SUCCESSFULL


sgemm:

SUCCESSFULL


softmax-output:

SUCCESSFULL


winograd:

SUCCESSFULL
