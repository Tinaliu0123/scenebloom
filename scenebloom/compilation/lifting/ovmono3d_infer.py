"""OVMono3D-LIFT inference as in upstream ``demo/demo.py``, returning per-detection data."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scenebloom.utils.paths import CHECKPOINTS, THIRD_PARTY

OVDIR = THIRD_PARTY / "ovmono3d"
# BERT (GroundingDINO's text encoder) and DINOv2 load by hub id from checkpoints/, offline;
# transformers reads the cache path at import time, so it is set before the imports.
os.environ["HF_HUB_CACHE"] = str(CHECKPOINTS / "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
# transformers prefers these legacy variables over HF_HUB_CACHE when they are set.
for _legacy in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"):
    os.environ.pop(_legacy, None)
torch.hub.set_dir(str(CHECKPOINTS / "torch_hub"))
sys.path.insert(0, str(OVDIR))
sys.dont_write_bytecode = True
np.set_printoptions(suppress=True)
# GroundingDINO in cubercnn opens "./configs/GroundingDINO_SwinB_cfg.py", so run from
# OVDIR; every path the CLI passes is absolute.
os.chdir(str(OVDIR))

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402,F401  (used by ovmono3d_cli)
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import transforms as T  # noqa: E402
from detectron2.engine import default_setup  # noqa: E402

from cubercnn.config import get_cfg_defaults  # noqa: E402
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa: E402,F401
from cubercnn.modeling.meta_arch import RCNN3D, build_model  # noqa: E402,F401
from cubercnn.modeling.proposal_generator import RPNWithIgnore  # noqa: E402,F401
from cubercnn.modeling.roi_heads import ROIHeads3D  # noqa: E402,F401
from cubercnn import util  # noqa: E402
from tqdm import tqdm  # noqa: E402


def _setup(args) -> object:
    cfg = get_cfg()
    get_cfg_defaults(cfg)
    config_file = args.config_file
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def _do_inference(args, cfg, model) -> dict:
    list_of_ims = util.list_files(os.path.join(args.input_folder, ''), '*')
    list_of_ims = [im for im in list_of_ims if not im.endswith('.json') and not im.endswith('.npz')]
    list_of_cats_per_img = util.load_json(args.labels_file)

    model.eval()
    thres = args.threshold

    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    predictions_per_image: dict[str, list[dict]] = {}

    for path in tqdm(list_of_ims):
        im_name = util.file_parts(path)[1]
        im = util.imread(path)
        if im is None:
            continue
        cats = list_of_cats_per_img.get(im_name, [])
        if not cats:
            continue
        h, w = im.shape[:2]

        # K resolution: NPZ overrides --focal-length if present (verbatim demo.py).
        focal_length = args.focal_length
        if focal_length == 0:
            focal_length_ndc = 4.0
            focal_length = focal_length_ndc * h / 2
        principal_point = args.principal_point
        if len(principal_point) == 0:
            px, py = w / 2, h / 2
        else:
            px, py = principal_point
        K = np.array([[focal_length, 0.0, px], [0.0, focal_length, py], [0.0, 0.0, 1.0]])

        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image

        depth_npz = os.path.join(args.input_folder, im_name + '.npz')
        depth_tensor = None
        if os.path.exists(depth_npz):
            import cv2
            npz = np.load(depth_npz)
            depth_np = npz['depth'].astype(np.float32)
            K = npz['K'].astype(np.float32)
            depth_resized = cv2.resize(depth_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            depth_tensor = torch.as_tensor(depth_resized).unsqueeze(0).cuda()

        batched_dict = {
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(),
            'height': im.shape[0], 'width': im.shape[1], 'K': K, 'category_list': cats,
        }
        if depth_tensor is not None:
            batched_dict['depth'] = depth_tensor
        batched = [batched_dict]
        dets = model(batched)[0]['instances']
        n_det = len(dets)

        kept: list[dict] = []

        if n_det > 0:
            for corners3D, center_cam, dimensions, pose, score, cat_idx in zip(
                    dets.pred_bbox3D, dets.pred_center_cam,
                    dets.pred_dimensions, dets.pred_pose, dets.scores, dets.pred_classes,
            ):
                if score < thres:
                    continue
                cat = cats[cat_idx]
                pose_arr = pose.tolist()
                yaw_rad = float(np.arctan2(pose_arr[0][2], pose_arr[0][0])) if (
                    isinstance(pose_arr, list) and len(pose_arr) >= 1 and len(pose_arr[0]) >= 3) else None
                kept.append({
                    "label": cat,
                    "score": float(score),
                    "center_cam": center_cam.tolist(),
                    "dimensions_w_l_h": dimensions.tolist(),
                    "yaw_rad": yaw_rad,
                    "bbox3d_corners": corners3D.tolist(),
                })

        print('File: {} with {} dets'.format(im_name, len(kept)))
        predictions_per_image[im_name] = kept

    return {"predictions_per_image": predictions_per_image}
