#!/bin/bash

echo "========================================"
echo "MedSegDiff 快速启动脚本"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

while true; do
    echo ""
    echo "请选择操作:"
    echo "[1] 训练 ISIC 模型"
    echo "[2] 训练 BRATS 模型"
    echo "[3] ISIC 数据集推理"
    echo "[4] BRATS 数据集推理"
    echo "[5] 评估 ISIC 结果"
    echo "[6] 启动 Visdom 可视化服务器"
    echo "[7] 检查环境"
    echo "[0] 退出"
    echo ""

    read -p "请输入选项 (0-7): " choice

    case $choice in
        1)
            echo ""
            echo "========================================"
            echo "开始训练 ISIC 模型"
            echo "========================================"
            echo ""
            read -p "请输入ISIC训练数据目录路径 (例如: ./data/ISIC/Train): " data_dir
            read -p "请输入输出目录路径 (例如: ./results/isic_train): " out_dir
            read -p "请输入batch_size (默认8): " batch_size
            batch_size=${batch_size:-8}

            python scripts/segmentation_train.py \
                --data_name ISIC \
                --data_dir "$data_dir" \
                --out_dir "$out_dir" \
                --image_size 256 \
                --num_channels 128 \
                --class_cond False \
                --num_res_blocks 2 \
                --num_heads 1 \
                --learn_sigma True \
                --use_scale_shift_norm False \
                --attention_resolutions 16 \
                --diffusion_steps 1000 \
                --noise_schedule linear \
                --rescale_learned_sigmas False \
                --rescale_timesteps False \
                --lr 1e-4 \
                --batch_size $batch_size

            echo ""
            echo -e "${GREEN}训练完成！模型保存在: $out_dir${NC}"
            read -p "按回车键继续..."
            ;;

        2)
            echo ""
            echo "========================================"
            echo "开始训练 BRATS 模型"
            echo "========================================"
            echo ""
            read -p "请输入BRATS训练数据目录路径 (例如: ./data/training): " data_dir
            read -p "请输入输出目录路径 (例如: ./results/brats_train): " out_dir
            read -p "请输入batch_size (默认8): " batch_size
            batch_size=${batch_size:-8}

            python scripts/segmentation_train.py \
                --data_name BRATS \
                --data_dir "$data_dir" \
                --out_dir "$out_dir" \
                --image_size 256 \
                --num_channels 128 \
                --class_cond False \
                --num_res_blocks 2 \
                --num_heads 1 \
                --learn_sigma True \
                --use_scale_shift_norm False \
                --attention_resolutions 16 \
                --diffusion_steps 1000 \
                --noise_schedule linear \
                --rescale_learned_sigmas False \
                --rescale_timesteps False \
                --lr 1e-4 \
                --batch_size $batch_size

            echo ""
            echo -e "${GREEN}训练完成！模型保存在: $out_dir${NC}"
            read -p "按回车键继续..."
            ;;

        3)
            echo ""
            echo "========================================"
            echo "ISIC 数据集推理"
            echo "========================================"
            echo ""
            read -p "请输入ISIC测试数据目录路径 (例如: ./data/ISIC/Test): " data_dir
            read -p "请输入输出目录路径 (例如: ./results/isic_pred): " out_dir
            read -p "请输入模型路径 (例如: ./results/isic_train/model.pt): " model_path
            read -p "是否使用DPM-Solver加速? (y/n, 默认n): " use_dpm
            read -p "集成样本数量 (默认5): " num_ensemble
            num_ensemble=${num_ensemble:-5}

            if [ "$use_dpm" = "y" ]; then
                python scripts/segmentation_sample.py \
                    --data_name ISIC \
                    --data_dir "$data_dir" \
                    --out_dir "$out_dir" \
                    --model_path "$model_path" \
                    --image_size 256 \
                    --num_channels 128 \
                    --class_cond False \
                    --num_res_blocks 2 \
                    --num_heads 1 \
                    --learn_sigma True \
                    --use_scale_shift_norm False \
                    --attention_resolutions 16 \
                    --diffusion_steps 20 \
                    --dpm_solver True \
                    --noise_schedule linear \
                    --rescale_learned_sigmas False \
                    --rescale_timesteps False \
                    --num_ensemble $num_ensemble
            else
                python scripts/segmentation_sample.py \
                    --data_name ISIC \
                    --data_dir "$data_dir" \
                    --out_dir "$out_dir" \
                    --model_path "$model_path" \
                    --image_size 256 \
                    --num_channels 128 \
                    --class_cond False \
                    --num_res_blocks 2 \
                    --num_heads 1 \
                    --learn_sigma True \
                    --use_scale_shift_norm False \
                    --attention_resolutions 16 \
                    --diffusion_steps 1000 \
                    --noise_schedule linear \
                    --rescale_learned_sigmas False \
                    --rescale_timesteps False \
                    --num_ensemble $num_ensemble
            fi

            echo ""
            echo -e "${GREEN}推理完成！结果保存在: $out_dir${NC}"
            read -p "按回车键继续..."
            ;;

        4)
            echo ""
            echo "========================================"
            echo "BRATS 数据集推理"
            echo "========================================"
            echo ""
            read -p "请输入BRATS测试数据目录路径 (例如: ./data/testing): " data_dir
            read -p "请输入输出目录路径 (例如: ./results/brats_pred): " out_dir
            read -p "请输入模型路径 (例如: ./results/brats_train/model.pt): " model_path
            read -p "集成样本数量 (默认5): " num_ensemble
            num_ensemble=${num_ensemble:-5}

            python scripts/segmentation_sample.py \
                --data_name BRATS \
                --data_dir "$data_dir" \
                --out_dir "$out_dir" \
                --model_path "$model_path" \
                --image_size 256 \
                --num_channels 128 \
                --class_cond False \
                --num_res_blocks 2 \
                --num_heads 1 \
                --learn_sigma True \
                --use_scale_shift_norm False \
                --attention_resolutions 16 \
                --diffusion_steps 1000 \
                --noise_schedule linear \
                --rescale_learned_sigmas False \
                --rescale_timesteps False \
                --num_ensemble $num_ensemble

            echo ""
            echo -e "${GREEN}推理完成！结果保存在: $out_dir${NC}"
            read -p "按回车键继续..."
            ;;

        5)
            echo ""
            echo "========================================"
            echo "评估 ISIC 结果"
            echo "========================================"
            echo ""
            read -p "请输入预测结果目录路径: " pred_dir
            read -p "请输入真值目录路径: " gt_dir

            python scripts/segmentation_env.py \
                --inp_pth "$pred_dir" \
                --out_pth "$gt_dir"

            echo ""
            echo -e "${GREEN}评估完成！${NC}"
            read -p "按回车键继续..."
            ;;

        6)
            echo ""
            echo "========================================"
            echo "启动 Visdom 服务器"
            echo "========================================"
            echo -e "${BLUE}服务器地址: http://localhost:8850${NC}"
            echo "按 Ctrl+C 停止服务器"
            echo ""
            python -m visdom.server -port 8850
            read -p "按回车键继续..."
            ;;

        7)
            echo ""
            echo "========================================"
            echo "检查环境配置"
            echo "========================================"
            echo ""

            echo "[1] 检查 Python 版本"
            python --version
            echo ""

            echo "[2] 检查 PyTorch 和 CUDA"
            python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU数量:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('GPU名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
            echo ""

            echo "[3] 检查依赖包"
            python -c "
packages = ['torch', 'numpy', 'pandas', 'nibabel', 'cv2', 'skimage', 'matplotlib', 'visdom', 'torchsummary']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} (未安装)')
"
            echo ""

            echo "[4] 检查数据集目录"
            if [ -d "data/ISIC" ]; then
                echo -e "${GREEN}✓ ISIC 数据集目录存在${NC}"
            else
                echo "✗ ISIC 数据集目录不存在，请下载并放置在 data/ISIC 目录"
            fi

            if [ -d "data/training" ]; then
                echo -e "${GREEN}✓ BRATS 训练数据目录存在${NC}"
            else
                echo "✗ BRATS 训练数据目录不存在，请下载并放置在 data/training 目录"
            fi
            echo ""

            echo "环境检查完成！"
            read -p "按回车键继续..."
            ;;

        0)
            echo ""
            echo "感谢使用 MedSegDiff！"
            exit 0
            ;;

        *)
            echo "无效选项，请重新选择"
            ;;
    esac
done
