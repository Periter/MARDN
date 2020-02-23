# python main.py --model "MARDN" --scale 2 --test_only --save_results --save "MARD_x2" --resume 0 --pre_train "../pre_train/MARDN_x2.pt"  --data_test "Set5+Set14+B100+Urban100+manga109"
python main.py --model "MARDN" --scale 4 --test_only --save_results --save "MARD_x4" --resume 0 --pre_train "../pre_train/MARDN_x4.pt"  --data_test "Set5+Set14+B100+Urban100+manga109"
# python main.py --model "MARDN" --scale 8 --test_only --save_results --save "MARD_x8" --resume 0 --pre_train "../pre_train/MARDN_x8.pt"  --data_test "Set5+Set14+B100+Urban100+manga109"
