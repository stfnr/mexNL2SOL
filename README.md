# mexNL2SOL
Mex wrapper for NL2SOL

NL2SOL MEX

Here you can find a very minimal MATLAB wrapper to NL2SOL (with bound constraints) from the PORT library. I have so far only tested it on Linux with MATLAB R2013b, but I think it should also function properly under windows and mac.

Installing is easy if you have a FORTRAN and C compiler installed and configured for compiling MEX files. Simply add the repo to your MATLAB path. Type compileNL2SOL, and you should be ready to go.

Syntax is similar to lsqnonlin. [X,RESNORM,RESIDUAL,EXITFLAG,ITERATIONS,FEVALS,JACOBIAN] = mexnl2sol(FUN,X0,(LB),(UB),(OPTIONS))

Note however that the output struct of lsqnonlin is replaced with iterations and lambda with the number of function evaluations.

If you have tested it on another MATLAB/OS combination, or find a bug in the code, please drop me a line, then I will update the code/wiki.

## Precompiled mex files

### MacOS

#### Apple Intel 

Make sure gcc10 (including gfortran) is installed. 

Requires /usr/local/opt/gcc/lib/gcc/10/libgfortran.5.dylib which can be installed using e.g. home brew. 

E.g. install gcc10 with brew 
> brew install gcc@10
And create the following symbolic link: 
> ln -s /usr/local/Cellar/gcc@10/10.3.0/lib/gcc/10/ /usr/local/opt/gcc/lib/gcc/10

//> ln -s /usr/local/Cellar/gcc/11.1.0_1/lib/gcc/11 /usr/local/opt/gcc/lib/gcc/11

#### Apple Silicon / arm64 

Make sure gcc12 (including gfortran) is installed. 
Requires /opt/homebrew/Cellar/gcc@12/12.3.0/bin/gfortran-12

Install gcc12 with brew 
> brew install gcc@12

And also create the following symbolic links: 
> ln -s /opt/homebrew/bin/gfortran-12 /opt/homebrew/bin/gfortran
> ln -s /opt/homebrew/Cellar/gcc@12/12.3.0/bin/gfortran-12 /opt/homebrew/Cellar/gcc@12/12.3.0/bin/gfortran

#### Frequent issues

Make sure commandline tools are installed by running the terminal:

> xcode-select --install

Error:  No supported compiler was found. For options, visit https://www.mathworks.com/support/compilers.

MacOS SDKs can be found here: 
https://github.com/phracker/MacOSX-SDKs