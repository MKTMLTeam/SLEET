@ECHO OFF
set HYDRA_FULL_ERROR=1
conda activate SLEET
python3 scripts/test.py
