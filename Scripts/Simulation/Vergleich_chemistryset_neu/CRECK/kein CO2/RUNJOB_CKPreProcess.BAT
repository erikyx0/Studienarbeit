REM Generated Batch Job Command File

REM get the runtime environment defined

CALL "E:\Ansys\ANSYS Inc\ANSYS Student\v252\reaction\chemkin.win64\bin\run_chemkin_env_setup.bat"
REM cd to the Working Directory for this job
C: && cd \Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\ATR\kein CO2
SET MKL_NUM_THREADS=1
SET OMP_NUM_THREADS=1
SET XERCES_DISABLE_DTD=1

REM RUN THE JOB AND CAPTURE EXIT STATUS

CKPreProcess.EXE -i "C:\Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\ATR\kein CO2\CKPreProc_ATR_neu22.input"
exit %errorlevel%
