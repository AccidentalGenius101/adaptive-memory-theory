@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=%PATH%;C:\Users\briel\AppData\Local\Programs\Python\Python312\Scripts
echo === CUDA check ===
C:\Users\briel\AppData\Local\Programs\Python\Python312\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('CUDA ver:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo === Compiling v4 ===
C:\Users\briel\AppData\Local\Programs\Python\Python312\python.exe -c "
import sys, os
sys.path.insert(0, r'C:\Users\briel\Documents\vcsm-theory')
os.chdir(r'C:\Users\briel\Documents\vcsm-theory')
import vcml_gpu_v4
vcml_gpu_v4.benchmark(L=80, B=8, nsteps=5000)
" 2>&1
