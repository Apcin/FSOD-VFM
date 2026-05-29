CUDA_VISIBLE_DEVICES=0 python ./main.py \
--json_path ./data/PascalVOC/vocsplit/split1/1shot_seed33.json \
--test_json ./data/PascalVOC/VOC2007Test/voc07test_coco_format.json \
--test_img_dir /mnt/data/wangzijian/object_detection_datasets/PascalVOC/VOC2007Test/VOC2007/JPEGImages \
--data_dir /mnt/data/wangzijian/object_detection_datasets \
--target_categories bus sofa cow bird motorbike \
--filter_by_categories \
--model_version  dinov2_vitl14 \
--feat_extractor_name DINOV2 \
--repo_or_dir /mnt/data/wangzijian/FSOD-VFM-Public/dinov2 \
--dinov2_checkpoint_dir  /mnt/data/wangzijian/FSOD-VFM-Public/checkpoints \
--min_threshold 0.01 \
--diffusion_steps 30 \
--alp 0.3 \
--lamb 0.5 

#target categories for split1: bus sofa cow bird motorbike; split2: horse aeroplane bottle sofa cow; split3: cat motorbike boat sofa sheep.
