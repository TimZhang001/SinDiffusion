"""
Train a diffusion model on images.
"""

import argparse
import os
import math
from PIL import Image
import torchvision as tv
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    adjust_scales2image
)
from guided_diffusion.train_util import TrainLoop, parse_resume_step_from_filename


mvtectAD = "/raid/zhangss/dataset/ADetection/mvtecAD/"
mvtectTexture = ["grid", "carpet", "leather", "tile", "wood"]
mvtectObject1 = ["hazelnut", "bottle", "cable", "capsule",  "pill"] # , 
mvtectObject2 = ["screw", "metal_nut", "pill", "toothbrush", "transistor", "zipper"] #  

def main():
    
    for target in mvtectTexture: 
        target_path = os.path.join(mvtectAD, target, "test")
        file_names  = os.listdir(target_path)
        for file_name in file_names:
            if file_name in ['good']:
                continue

            args = create_argparser().parse_args()

            # 自定义修改参数
            #args.data_dir        = '/home/zhangss/PHDPaper/04_SinDiffusion/dataset/mvtec/hazelnut/004.png'
            args.data_dir        = os.path.join(target_path, file_name, "000.png")
            args.lr              = 5e-4 / 8
            args.diffusion_steps = 1000
            args.image_size      = 256
            args.batch_size      = 16
            args.noise_schedule  = 'linear'
            
            args.num_channels    = 64
            args.channel_mult    = "1,2,4"
            args.attention_resolutions = "2"
            args.num_res_blocks  = 1
            args.resblock_updown = False
            
            args.use_fp16        = True
            args.use_scale_shift_norm = True 
            args.use_checkpoint  = True 
            
            args.save_dir        = '/home/zhangss/PHDPaper/04_SinDiffusion/RESULT/'
            args.lr_anneal_steps = args.lr_anneal_steps / 4
            args.save_interval   = args.save_interval / 8
            args.save_dir        = os.path.join(args.save_dir, target, file_name)

            dist_util.setup_dist()
            logger.configure(dir=args.save_dir)

            # 读取图片
            real = Image.open(args.data_dir).resize((args.image_size, args.image_size), Image.LANCZOS).convert('RGB')
            real = tv.transforms.ToTensor()(real)[None]

            adjust_scales2image(real, args)

            logger.configure(dir=args.save_dir)

            logger.log("creating model and diffusion...")
            model, diffusion = create_model_and_diffusion(
                **args_to_dict(args, model_and_diffusion_defaults().keys())
            )
            model.to(dist_util.dev())
            schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

            logger.log("creating data loader...")
            data = load_data(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                image_size=args.image_size,
                class_cond=args.class_cond,
                random_crop=False,
                random_flip=False,
                scale_init=args.scale1,
                scale_factor=args.scale_factor,
                stop_scale=args.stop_scale,
                current_scale=args.stop_scale
            )

            logger.log("training...")
            TrainLoop(
                model=model,
                diffusion=diffusion,
                data=data,
                batch_size=args.batch_size,
                microbatch=args.microbatch,
                lr=args.lr,
                ema_rate=args.ema_rate,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                resume_checkpoint=args.resume_checkpoint,
                use_fp16=args.use_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
                schedule_sampler=schedule_sampler,
                weight_decay=args.weight_decay,
                lr_anneal_steps=args.lr_anneal_steps,
            ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=50000,
        num_channels_init=128,
        num_res_blocks_init=6,
        scale_factor_init=0.75,
        min_size=25,
        max_size=250,
        nc_im=3,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
