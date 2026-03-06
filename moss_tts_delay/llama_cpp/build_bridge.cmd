@echo off
setlocal

REM Build the C bridge shared library for the llama.cpp backbone using MSVC.
REM Run this from within the Visual Studio Developer Command Prompt.

set "SCRIPT_DIR=%~dp0"
set "BRIDGE_SRC=%SCRIPT_DIR%backbone_bridge.c"
set "LIB_OUT=%SCRIPT_DIR%libbackbone_bridge.dll"

REM We expect llama.cpp to be located at o:\voc\llama.cpp based on your recent commands
set "LLAMA_CPP_DIR=o:\voc\llama.cpp"

if not exist "%LLAMA_CPP_DIR%" (
    echo Error: Could not find llama.cpp at %LLAMA_CPP_DIR%
    echo Please edit this script and set LLAMA_CPP_DIR to the correct path.
    exit /b 1
)

set "INCLUDE_DIRS=/I"%LLAMA_CPP_DIR%\include" /I"%LLAMA_CPP_DIR%\ggml\include""

REM The compiled import library (.lib) is usually created in the build\bin\Release or build\src\Release directory
set "LIB_DIR="
if exist "%LLAMA_CPP_DIR%\build\bin\Release\llama.lib" set "LIB_DIR=%LLAMA_CPP_DIR%\build\bin\Release"
if exist "%LLAMA_CPP_DIR%\build\src\Release\llama.lib" set "LIB_DIR=%LLAMA_CPP_DIR%\build\src\Release"
if exist "%LLAMA_CPP_DIR%\build\src\llama.lib" set "LIB_DIR=%LLAMA_CPP_DIR%\build\src"
if exist "%LLAMA_CPP_DIR%\build\bin\llama.lib" set "LIB_DIR=%LLAMA_CPP_DIR%\build\bin"

if "%LIB_DIR%"=="" (
    echo Error: Could not find llama.lib in %LLAMA_CPP_DIR%\build\src or its Release subfolders.
    echo Make sure you have successfully compiled llama.cpp with: cmake --build build --config Release
    exit /b 1
)

echo Bridge source:  %BRIDGE_SRC%
echo llama.cpp dir:  %LLAMA_CPP_DIR%
echo Library dir:    %LIB_DIR%
echo Output:         %LIB_OUT%
echo.

REM Compile directly using MSVC cl.exe
cl.exe /nologo /LD /O2 /EHsc %INCLUDE_DIRS% "%BRIDGE_SRC%" /link /LIBPATH:"%LIB_DIR%" llama.lib /OUT:"%LIB_OUT%"

if %errorlevel% neq 0 (
    echo.
    echo Compilation failed.
    exit /b %errorlevel%
)

echo.
echo Successfully built: %LIB_OUT%
echo You can now run the MOSS TTS GGUF backend.

REM Clean up intermediate object file
if exist "backbone_bridge.obj" del "backbone_bridge.obj"
if exist "%SCRIPT_DIR%libbackbone_bridge.exp" del "%SCRIPT_DIR%libbackbone_bridge.exp"
if exist "%SCRIPT_DIR%libbackbone_bridge.lib" del "%SCRIPT_DIR%libbackbone_bridge.lib"

endlocal
