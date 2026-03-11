@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set PATH=%PATH%;C:\Users\briel\AppData\Local\Programs\Python\Python312\Scripts
cd /d C:\Users\briel\Documents\vcsm-theory
py vcml_gpu_v4.py --benchmark
