@echo off
cd /d c:\Users\ASUS\Documents\AITF-2026\PKL\Preprocessing\hybrid_preprocessing
"C:\Users\ASUS\AppData\Local\Microsoft\WindowsApps\python3.11.exe" run_preprocessing.py --input "..\..\Datasets\main_review.csv" --output-dir "..\..\Datasets\main_review_preprocessed" --chunksize 2000
if %errorlevel% neq 0 echo ERROR %errorlevel%
echo DONE
