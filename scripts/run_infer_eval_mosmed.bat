@echo off
REM ============================================================
REM  MosMed inference + evaluation (step 73000 checkpoint)
REM  Usage:
REM    1. Place MosMed data under repo root (see README_MOSMED.md)
REM    2. Place trained checkpoint at:
REM       scripts/results/mosmed_bcedice_v2/savedmodel073000.pt
REM    3. Run this script from repo root:
REM       scripts\run_infer_eval_mosmed.bat
REM ============================================================

REM Resolve repo root from script location (parent of scripts/)
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

set "DATA_ROOT=%REPO_ROOT%\MosMedData-Chest-CT-Scans-with-COVID-19-Related-Findings-main"
set "MODEL_PATH=scripts\results\mosmed_bcedice_v2\savedmodel073000.pt"
set "PRED_DIR=scripts\results\mosmed_pred_v2_73k"

echo ============================================================
echo  [1/2] Inference 20 slices (num_ensemble=1, quick test)
echo ============================================================
python scripts\segmentation_sample.py ^
    --data_name MOSMED ^
    --data_dir "%DATA_ROOT%" ^
    --image_size 256 ^
    --num_channels 128 ^
    --in_ch 2 ^
    --version new ^
    --loss_type bce_dice ^
    --bce_pos_weight 200 ^
    --tversky_alpha 0.3 ^
    --tversky_beta 0.7 ^
    --batch_size 1 ^
    --num_ensemble 1 ^
    --model_path "%MODEL_PATH%" ^
    --out_dir "%PRED_DIR%" ^
    --max_samples 20

if errorlevel 1 (
    echo.
    echo [ERROR] Inference failed, check error messages above
    popd
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  [2/2] Evaluate Dice / IoU
echo ============================================================
python scripts\mosmed_eval.py ^
    --pred_dir "%PRED_DIR%" ^
    --data_root "%DATA_ROOT%" ^
    --image_size 256

echo.
echo ============================================================
echo  Done! Mean Dice / IoU shown above
echo  Expected: mean Dice ~0.29 (step 73000 checkpoint)
echo ============================================================
popd
pause
