export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=0
python eval_xai.py checkpoints/checkpoint_best_xai_apex_M2P_1e-3_base_released.pt.pt  xai_data_bin_apex_reg_cls/0 --task xai_M2P
