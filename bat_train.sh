mpiexec -n 8 python image_train.py --data_dir data/yourimage.png --lr 5e-4 --diffusion_steps 1000 --image_size 256
                                   --noise_schedule linear --num_channels 64 --num_head_channels 16 --channel_mult "1,2,4" 
                                   --attention_resolutions "2" --num_res_blocks 1 --resblock_updown False --use_fp16 True 
                                   --use_scale_shift_norm True --use_checkpoint True --batch_size 16