REM Generated Batch Job Command File

REM get the runtime environment defined

CALL "E:\Ansys\ANSYS Inc\ANSYS Student\v252\reaction\chemkin.win64\bin\run_chemkin_env_setup.bat"
REM cd to the Working Directory for this job
C: && cd \Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\GRI\kein CO2
SET MKL_NUM_THREADS=1
SET OMP_NUM_THREADS=1
SET XERCES_DISABLE_DTD=1

REM DELETE ANY INTERMEDIATE FILES LEFTOVER FROM PREVIOUS RUN

IF EXIST "C:\Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\GRI\kein CO2\XMLdata_2_Gas_MixerC3.zip" DEL /F /Q "C:\Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\GRI\kein CO2\XMLdata_2_Gas_MixerC3.zip"

REM COPY FILES TO WORKING DIR

COPY /Y "E:\Ansys\ANSYS Inc\ANSYS Student\v252\reaction\chemkin.win64\data\chemkindata.dtd" "C:\Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\GRI\kein CO2"

REM RUN THE JOB AND CAPTURE EXIT STATUS


CKReactorMixer -i "C:\Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\GRI\kein CO2\2_c3.inp" -o "C:\Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\GRI\kein CO2\2_c3.out" -x "C:\Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\GRI\kein CO2\XMLdata_2_Gas_MixerC3.zip" -m "2_c3.out.mon" -c "C:\Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\GRI\kein CO2\GriMech_gas.asc" -t "C:\Users\erikd\Desktop\Studienarbeit\Scripts\Simulation\Vergleich_chemistryset_neu\GRI\kein CO2\GriMech_gtran.asc"
set exit_status=%errorlevel%
if exist CKRUN_ERROR.TXT del /F /Q CKRUN_ERROR.TXT
if exist CKRUN_SUCCESS.TXT del /F /Q CKRUN_SUCCESS.TXT
if %exit_status% gtr 0 echo exit_status=%exit_status% >CKRUN_ERROR.TXT
if %exit_status% equ 0 echo exit_status=%exit_status% >CKRUN_SUCCESS.TXT
exit %exit_status%
