python image_sample.py --data_dir dataset/mvtec/hazelnut/004.png --diffusion_steps 1000 --image_size 256 --noise_schedule linear \
                       --num_channels 64 --num_head_channels 16 --num_res_blocks 1 --channel_mult "1,2,4" \
                       --attention_resolution "2" --resblock_updown False --use_fp16 True --use_scale_shift_norm True  \
                       --use_checkpoint True --model_root output \
                       --results_path RESULT/