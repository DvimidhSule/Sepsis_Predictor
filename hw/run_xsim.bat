@echo off
setlocal enabledelayedexpansion

REM Run the Sepsis Predictor hardware co-simulations in Vivado XSim.

REM Check if xvlog is in path
where xvlog >nul 2>nul
if %errorlevel% equ 0 goto :path_ok

echo Vivado tools not found in PATH. Attempting to locate Vivado 2026.1...
if exist "d:\VIVADO\2026.1\Vivado\bin" (
    set "PATH=d:\VIVADO\2026.1\Vivado\bin;%PATH%"
    echo Added d:\VIVADO\2026.1\Vivado\bin to PATH.
    goto :path_ok
)
if exist "C:\Xilinx\Vivado\2026.1\bin" (
    set "PATH=C:\Xilinx\Vivado\2026.1\bin;%PATH%"
    echo Added C:\Xilinx\Vivado\2026.1\bin to PATH.
    goto :path_ok
)
if exist "d:\VIVADO\Vivado\2024.1\bin" (
    set "PATH=d:\VIVADO\Vivado\2024.1\bin;%PATH%"
    echo Added d:\VIVADO\Vivado\2024.1\bin to PATH.
    goto :path_ok
)
echo WARNING: Vivado tools were not found in common locations.
echo If commands fail, please ensure Vivado is installed and in your PATH.

:path_ok

set "OPT=%~1"
if not "!OPT!"=="" goto :run_selected

echo ========================================================
echo Sepsis Predictor Vivado XSim Simulation Launcher
echo ========================================================
echo Select simulation mode:
echo [1] Run co-simulation verification (tb_sepsis_engine) - Batch Mode
echo [2] Run patient demo simulation (tb_demo_patients) - Batch Mode
echo [3] Run patient demo simulation (tb_demo_patients) - GUI Mode (auto-waves)
echo ========================================================
echo.
set /p OPT="Enter choice (1-3): "

:run_selected
if "!OPT!"=="1" goto :cosim
if "!OPT!"=="2" goto :patient_batch
if "!OPT!"=="3" goto :patient_gui
echo Invalid choice: !OPT!
goto :err

:cosim
echo.
echo === Compiling tb_sepsis_engine ===
call xvlog hw\sepsis_engine.v hw\tb_sepsis_engine.v
if errorlevel 1 goto :err

echo === Elaborating tb_sepsis_engine ===
call xelab tb_sepsis_engine -s tb_sim -debug typical
if errorlevel 1 goto :err

echo === Simulating (co-sim vs Python golden vectors) ===
call xsim tb_sim -runall
if errorlevel 1 goto :err
goto :done

:patient_batch
echo.
echo === Compiling tb_demo_patients ===
call xvlog hw\sepsis_engine.v hw\tb_demo_patients.v
if errorlevel 1 goto :err

echo === Elaborating tb_demo_patients ===
call xelab tb_demo_patients -s tb_demo_sim -debug typical
if errorlevel 1 goto :err

echo === Simulating (patient demo batch) ===
call xsim tb_demo_sim -runall
if errorlevel 1 goto :err
goto :done

:patient_gui
echo.
echo === Compiling tb_demo_patients ===
call xvlog hw\sepsis_engine.v hw\tb_demo_patients.v
if errorlevel 1 goto :err

echo === Elaborating tb_demo_patients ===
call xelab tb_demo_patients -s tb_demo_sim -debug typical
if errorlevel 1 goto :err

echo === Starting GUI Waveform Viewer (tb_demo_patients) ===
call xsim tb_demo_sim -gui -tclbatch hw/tb_demo_patients.tcl
if errorlevel 1 goto :err
goto :done

:done
echo.
echo Simulation task completed successfully.
exit /b 0

:err
echo.
echo BUILD/SIM FAILED -- check messages above.
exit /b 1
