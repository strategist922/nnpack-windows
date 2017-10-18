# nnpack-windows
nnpack for Windows

Windows port of Marat Dukhan NNPACK - BSD 2-Clause "Simplified" (https://github.com/Maratyszcza/NNPACK)

The steps to build the nnpack-windows repo:

(Check you have Visual Studio 2017 and Python installed)

First you have to install PeachPy.
Open a Phyton command prompt with Administrator rights and type:

pip install --upgrade git+https://github.com/Maratyszcza/PeachPy

Download the nnpack-windows repo somewhere and after extracting it 
go in the command prompt to the directory ..\nnpack-windows\src\x86_64-fma and type:

buildme.bat

After all the objects are created you open te visual studio solution file with vs2017

(In the buildme.bat script you can specify your target cpu by changing -mcp=haswell to -mcp=skylake for example)

