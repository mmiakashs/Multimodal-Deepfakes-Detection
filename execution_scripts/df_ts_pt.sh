python3 ../deepfake_training.py \
-lr 0.0003 \
-ng 1 \
-cdn 1 \
-sml 9 \
-bs 16 \
-ep 500 \
-enl 2 \
-fes 256 \
-lhs 256 \
-menh 2 \
-mmnh 2 \
-lld 0.3 \
-uld 0.3 \
-cout 64 \
-mmattn_type 'concat' \
-logbd 'log' \
-logf 'exe_dfts_st_pt_h22_dp3.log' \
-tbl \
-tb_wn 'tb_runs/tb_dfts_st_pt/h22_dp3' \
-mcp 'dfts_pt_' \
-lstm_bi \
-dfp '/data/research_data/dfdc_embed_small' \
-is_guiding \
-cm 2 \
-cl 50 \
-ipt \
-ist
