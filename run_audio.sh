
########################################## Train: Image Output ##########################################
# 2d, depth: pixel out: image
screen -S train1 -dm bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py ./configs/small_cuboid/2d_out_img_1/config.yaml; \
                              CUDA_VISIBLE_DEVICES=0 python main.py ./configs/small_cuboid/2d_out_img_2/config.yaml; \
                              CUDA_VISIBLE_DEVICES=0 python main.py ./configs/small_cuboid/2d_out_img_3/config.yaml; \
                              exec sh';

########################################## Evaluation: Image Output ##########################################
# 2d, depth: pixel out: image evaluation
screen -S eval1 -dm bash -c 'CUDA_VISIBLE_DEVICES=0 python eval.py ./configs/small_cuboid/2d_out_img_1/config.yaml ./logs_True_False_False_image_conv2d-encoder-decoder_True_pixel_1/lightning_logs/checkpoints; \
                             CUDA_VISIBLE_DEVICES=0 python eval.py ./configs/small_cuboid/2d_out_img_2/config.yaml ./logs_True_False_False_image_conv2d-encoder-decoder_True_pixel_2/lightning_logs/checkpoints; \
                             CUDA_VISIBLE_DEVICES=0 python eval.py ./configs/small_cuboid/2d_out_img_3/config.yaml ./logs_True_False_False_image_conv2d-encoder-decoder_True_pixel_3/lightning_logs/checkpoints; \
                             exec sh';




########################################## Train: Depth L1 Output ##########################################
# 2d, depth: pixel out: depth
screen -S train5 -dm bash -c 'CUDA_VISIBLE_DEVICES=4 python main.py ./configs/small_cuboid/2d_out_depth_1/config.yaml; \
                              CUDA_VISIBLE_DEVICES=4 python main.py ./configs/small_cuboid/2d_out_depth_2/config.yaml; \
                              CUDA_VISIBLE_DEVICES=4 python main.py ./configs/small_cuboid/2d_out_depth_3/config.yaml; \
                              exec sh';


########################################## Evaluation: Depth L1 Output ##########################################
# 2d, depth: pixel out: depth evaluation
screen -S eval5 -dm bash -c 'CUDA_VISIBLE_DEVICES=4 python eval.py ./configs/small_cuboid/2d_out_depth_1/config.yaml ./logs_True_False_False_image_conv2d-encoder-decoder_True_depth-l1_1/lightning_logs/checkpoints; \
                             CUDA_VISIBLE_DEVICES=4 python eval.py ./configs/small_cuboid/2d_out_depth_2/config.yaml ./logs_True_False_False_image_conv2d-encoder-decoder_True_depth-l1_2/lightning_logs/checkpoints; \
                             CUDA_VISIBLE_DEVICES=4 python eval.py ./configs/small_cuboid/2d_out_depth_3/config.yaml ./logs_True_False_False_image_conv2d-encoder-decoder_True_depth-l1_3/lightning_logs/checkpoints; \
                             exec sh';