import argparse
import json
import torch
import torch.nn.functional
import model.dinov2
import model.sam2
import model.radio
import support_util
import query_util
import metric
import race_metric
import os
import hashlib
from chatrex.upn import UPNWrapper


def _file_identity(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {'path': path, 'exists': False}
    stat = os.stat(path)
    return {
        'path': path,
        'exists': True,
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def _build_support_cache_paths(args):
    _, sam2_checkpoint = model.sam2.get_sam2_model_cfg_and_ckpt_path(
        args.sam2_model_type
    )
    metadata = {
        'cache_version': 1,
        'feature_extractor': args.feat_extractor_name,
        'model_version': args.model_version,
        'radio_model_version': args.radio_model_version,
        'dinov2_image_size': args.dinov2_image_size,
        'sam2_model_type': args.sam2_model_type,
        'dinov2_checkpoint': _file_identity(args.pretrained),
        'sam2_checkpoint': _file_identity(sam2_checkpoint),
    }
    config_hash = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode('utf-8')
    ).hexdigest()
    with open(args.json_path, 'rb') as file:
        support_hash = hashlib.sha256(file.read()).hexdigest()
    support_name = os.path.splitext(os.path.basename(args.json_path))[0]
    instance_cache_path = os.path.join(
        args.support_feature_cache_dir,
        f"support_instances_{config_hash[:16]}.pt",
    )
    prototype_cache_path = os.path.join(
        args.prototype_cache_dir,
        f"{support_name}_{support_hash[:16]}_{config_hash[:16]}.pt",
    )
    metadata['support_json'] = os.path.abspath(args.json_path)
    metadata['support_hash'] = support_hash
    return metadata, instance_cache_path, prototype_cache_path

def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot VOC evaluation with DINOv2 + SAM2")

    parser.add_argument('--json_path', type=str,
                        default='./data/PascalVOC/vocsplit/seed0/box_10shot_train.json',
                        help='Path to support set JSON file (default: %(default)s)')

    parser.add_argument(
                        '--feat_extractor_name',
                        type=str,
                        default='DINOV2',
                        choices=['DINOV2', 'RADIO'],
                        help='feature extractor name (default: %(default)s)')

    parser.add_argument(
                        '--model_version',
                        type=str,
                        default='dinov2_vitl14',
                        choices=[
                            'dinov2_vits14', 'dinov2_vits14_reg',
                            'dinov2_vitb14', 'dinov2_vitb14_reg',
                            'dinov2_vitl14', 'dinov2_vitl14_reg',
                            'dinov2_vitg14', 'dinov2_vitg14_reg',
                        ],
                        help='model version (default: %(default)s)')

    parser.add_argument('--repo_or_dir', type=str,
                        default="/mnt/data/wangzijian/FSOD-VFM-Public/dinov2",
                        help='Repo or directory for dinov2 code (default: %(default)s)')

    parser.add_argument('--dinov2_checkpoint_dir', type=str,
                        default="/mnt/data/wangzijian/FSOD-VFM-Public/checkpoints",
                        help='Directory to pretrained dinov2 checkpoint (default: %(default)s)')

    parser.add_argument('--radio_model_version', type=str,
                        default='c-radio_v4-h',
                        help='RADIO model version when feat_extractor_name=RADIO (default: %(default)s)')

    parser.add_argument('--radio_cache_root', type=str,
                        default='./model_cache',
                        help='RADIO/model cache root (torch_hub under it) when feat_extractor_name=RADIO (default: %(default)s)')

    parser.add_argument('--sam2_model_type', type=str,
                        default='large',
                        help='SAM2 model type (small/medium/large) (default: %(default)s)')

    parser.add_argument('--data_dir', type=str,
                        default='./data/',
                        help='Root directory for dataset (default: %(default)s)')

    parser.add_argument('--dinov2_image_size', type=int,
                        default=630,
                        help='Input size for dinov2 images (default: %(default)s)')

    parser.add_argument('--test_json', type=str,
                        default='./data/PascalVOC/VOC2007Test/voc_split1.json',
                        help='COCO format test json (default: %(default)s)')

    parser.add_argument('--test_img_dir', type=str,
                        default='./data/coco/val2017',
                        help='Directory containing test images (default: %(default)s)')

    parser.add_argument('--pred_json', type=str,
                        default='temp_pred.json',
                        help='Output prediction JSON file (default: %(default)s)')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run models on (default: %(default)s)')
    
    parser.add_argument('--target_categories', type=str,nargs='+',
                        default=['bus','sofa','cow','bird','motorbike'],
                        help='Target categories for evaluation (default: %(default)s)')

    parser.add_argument('--min_threshold', type=float, default=0.01,
                        help='mean threshold for upn')

    parser.add_argument('--filter_by_categories', action='store_true',
                        help='filter by categories')

    parser.add_argument('--diffusion_steps', type=int, 
                        help='number of diffusion steps')

    parser.add_argument('--points_per_side', type=int,
                        default=32,
                        help='Points per side for SAM2 mask generator (default: %(default)s)')

    parser.add_argument('--alp', type=float, 
                        help='alpha in diffusion')
                        
    parser.add_argument('--lamb', type=float, 
                        help='lamda for decay')

    parser.add_argument('--race_eval', action='store_true',
                        help='evaluate predictions with competition protocol V1.5')
    parser.add_argument('--race_score_threshold', type=float, default=None,
                        help='optional score threshold for competition evaluation')
    parser.add_argument('--race_eval_output_dir', type=str, default='./results',
                        help='directory for competition evaluation summaries')
    parser.add_argument('--skip_coco_eval', action='store_true',
                        help='save predictions without running the original COCOeval')

    parser.add_argument('--prototype_cache_dir', type=str,
                        default='./cache/prototypes',
                        help='directory for final class prototype caches')
    parser.add_argument('--support_feature_cache_dir', type=str,
                        default='./cache/support_features',
                        help='directory for reusable support instance features')
    parser.add_argument('--disable_prototype_cache', action='store_true',
                        help='do not load or save final prototype caches')
    parser.add_argument('--disable_support_feature_cache', action='store_true',
                        help='do not load or save support instance features')
    parser.add_argument('--rebuild_prototype_cache', action='store_true',
                        help='recompute final prototypes (instance cache may still be reused)')
    parser.add_argument('--rebuild_support_feature_cache', action='store_true',
                        help='recompute support instance features and final prototypes')
    parser.add_argument('--support_box_batch_size', type=int, default=64,
                        help='number of support boxes processed together by SAM2')

    return parser.parse_args()

def main():
    args = parse_args()
    # Load UPN model if using UPN
    upn = None
    
    print('Loading UPN...')
    ckpt_path = '/mnt/data/wangzijian/FSOD-VFM-Public/checkpoints/upn_large.pth'
    upn = UPNWrapper(ckpt_path)

    model_base_names = [
        'dinov2_vits14',
        'dinov2_vitb14', 
        'dinov2_vitl14',
        'dinov2_vitg14',
    ]

    model_name = args.model_version
    is_reg = model_name.endswith('_reg')

    # Remove '_reg' to get the base name
    base_name = model_name.replace('_reg', '')

    if base_name in model_base_names:
        suffix = 'reg4_pretrain.pth' if is_reg else 'pretrain.pth'
        checkpoint_filename = f"{base_name}_{suffix}"
        args.pretrained = f"{args.dinov2_checkpoint_dir}/{checkpoint_filename}"
    else:
        # For models not in the base names, construct a default path
        suffix = 'reg4_pretrain.pth' if is_reg else 'pretrain.pth'
        checkpoint_filename = f"{base_name}_{suffix}"
        args.pretrained = f"{args.dinov2_checkpoint_dir}/{checkpoint_filename}"
    
    if args.feat_extractor_name == 'DINOV2':
        print('Loading Dinov2...')
        feat_extractor, image_transform = model.dinov2.load_dinov2_model(
            args.device,
            args.model_version,
            image_size=(args.dinov2_image_size, args.dinov2_image_size),
            repo_or_dir=args.repo_or_dir,
            pretrained=args.pretrained
        )
    elif args.feat_extractor_name == 'RADIO':
        print('Loading RADIO (C-RADIO v4-H)...')
        feat_extractor, image_transform = model.radio.load_radio_model(
            args.device,
            model_version=args.radio_model_version,
            cache_root=args.radio_cache_root,
            source='local',
        )
    else:
        raise ValueError(f"Unsupported feat_extractor_name: {args.feat_extractor_name}")

    print('Loading SAM2...')
    sam2_model, sam2_predictor, sam2_mask_generator = model.sam2.load_sam2_components(
        model_type=args.sam2_model_type,
        device=args.device,
        points_per_side=args.points_per_side
    )
        
    # Load support set
    with open(args.json_path, 'r') as f:
        support_data = json.load(f)

    # Print stats
    for cls, instances in support_data.items():
        print(f"Class: {cls}, #Instances: {len(instances)}")

    cache_metadata, instance_cache_path, prototype_cache_path = (
        _build_support_cache_paths(args)
    )
    use_prototype_cache = not args.disable_prototype_cache
    use_instance_cache = not args.disable_support_feature_cache
    force_prototype_rebuild = (
        args.rebuild_prototype_cache or args.rebuild_support_feature_cache
    )

    if (
        use_prototype_cache
        and not force_prototype_rebuild
        and os.path.exists(prototype_cache_path)
    ):
        prototype_cache = torch.load(
            prototype_cache_path, map_location='cpu', weights_only=False
        )
        proto_feat = prototype_cache['proto_feat'].to(args.device)
        proto_cls = prototype_cache['proto_cls']
        print(f"Loaded prototype cache: {prototype_cache_path}")
    else:
        # Build the memory bank. Multiple boxes from the same image reuse one
        # DINO/RADIO feature map and one SAM2 image embedding.
        memory_bank = support_util.extract_support_features(
            support_data,
            sam2_predictor,
            args.feat_extractor_name,
            feat_extractor,
            image_transform,
            args.data_dir,
            args.device,
            instance_cache_path=(instance_cache_path if use_instance_cache else None),
            rebuild_instance_cache=args.rebuild_support_feature_cache,
            cache_metadata=cache_metadata,
            support_box_batch_size=args.support_box_batch_size,
        )
        proto_feat, proto_cls = support_util.compute_prototype_weights(
            memory_bank, args.device
        )
        if use_prototype_cache:
            os.makedirs(args.prototype_cache_dir, exist_ok=True)
            temp_cache_path = f"{prototype_cache_path}.tmp"
            torch.save(
                {
                    'metadata': cache_metadata,
                    'proto_feat': proto_feat.detach().cpu(),
                    'proto_cls': proto_cls,
                },
                temp_cache_path,
            )
            os.replace(temp_cache_path, prototype_cache_path)
            print(f"Saved prototype cache: {prototype_cache_path}")
    min_th = 0.01
    
    # Load VOC2007 test loader
    image_paths, coco_style_loader = query_util.load_voc2007_coco_json(
        args.test_json,
        args.test_img_dir
    )

    # Generate predictions

    results = metric.generate_coco_style_predictions_upn(
        coco_style_loader,
        args.test_img_dir,
        sam2_predictor,
        args.feat_extractor_name,
        feat_extractor,
        image_transform,
        proto_feat,
        proto_cls,
        upn,  # Pass UPN model as parameter
        args.diffusion_steps,
        args.alp,
        args.lamb,
        args.device,
        args.min_threshold,
    )

    # Save predictions and optionally run the original COCO evaluation.
    if args.skip_coco_eval:
        metric.save_coco_predictions(results, args.pred_json)
        print(f"Predictions saved to: {args.pred_json}")
    else:
        metric.run_coco_eval(args.test_json, results, args.pred_json,target_categories=args.target_categories,filter_by_categories=args.filter_by_categories)

    if args.race_eval:
        race_metric.run_race_eval(
            gt_json_path=args.test_json,
            prediction_results=results,
            pred_json_path=args.pred_json,
            score_threshold=args.race_score_threshold,
            output_dir=args.race_eval_output_dir,
        )


if __name__ == '__main__':
    main()
