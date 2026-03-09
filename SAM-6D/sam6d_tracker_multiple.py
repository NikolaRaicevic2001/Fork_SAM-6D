#!/usr/bin/env python3
"""
sam6d_tracker_multiple.py

Run SAM-6D (ISM + PEM) for MULTIPLE objects in one RealSense stream.

Example:
python sam6d_tracker_multiple.py \
  --segmentor_model fastsam \
  --output_dir "$ROOT/Data/myObject/outputs" \
  --obj_name dominoSugar --cad_path "$ROOT/Data/myObject/dominoSugar/dominoSugar.ply" \
  --obj_name tomatoSoup  --cad_path "$ROOT/Data/myObject/tomatoSoup/tomatoSoup.ply" \
  --visualize

Notes / expectations:
- Each object needs its own templates dir containing:
    rgb_*.png, mask_*.png, xyz_*.npy
  Default: <output_dir>/<obj_name>/templates
  You can override per-object templates dir with:
    --templates_dir <path> (repeatable, aligned with obj_name/cad_path)

- This script shares the heavy ISM models (segmentor + descriptor) across objects.
- PEM network is shared; each object has its own extracted template features + model points.

- UDP output includes object name in payload ("obj_name").
"""

import os
import cv2
import sys
import time
import json
import glob
import torch
import socket
import random
import logging
import argparse
import importlib
import distinctipy
import numpy as np
import os.path as osp
import pycocotools.mask as cocomask

from PIL import Image
from skimage.feature import canny
from skimage.morphology import binary_dilation
from hydra.utils import instantiate
from hydra import initialize, compose
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from segment_anything.utils.amg import rle_to_mask

import gorilla
import trimesh

# =========================
# Repository paths (same layout you have)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISM_ROOT = os.path.join(BASE_DIR, "Instance_Segmentation_Model")
PEM_ROOT = os.path.join(BASE_DIR, "Pose_Estimation_Model")
CKPT_ROOT = os.path.join(ISM_ROOT, "checkpoints")

if ISM_ROOT not in sys.path:
    sys.path.insert(0, ISM_ROOT)
sys.path.append(os.path.join(PEM_ROOT, "provider"))
sys.path.append(os.path.join(PEM_ROOT, "utils"))
sys.path.append(os.path.join(PEM_ROOT, "model"))
sys.path.append(os.path.join(PEM_ROOT, "model", "pointnet2"))

from camera import RealSenseCamera
from draw_utils import draw_detections
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad
from Instance_Segmentation_Model.utils.inout import save_json_bop23
from Instance_Segmentation_Model.model.utils import Detections
from data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose
)
from Instance_Segmentation_Model.utils.poses.pose_utils import (
    get_obj_poses_from_template_level,
    load_index_level_in_level2
)

logging.basicConfig(level=logging.INFO)

rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# Helper Functions
# =========================
def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S
    q /= (np.linalg.norm(q) + 1e-12)
    return q

def decode_coco_segmentation(seg):
    """
    Decode COCO-style segmentation to a boolean mask, robust to SAM-6D/FastSAM quirks:
    - seg["size"] may be [H,W], [1,H,W], [H,W,1], etc.
    - seg["counts"] may be a list containing ints/strings (uncompressed RLE)
    - seg may be [dict] in some pipelines
    """
    if seg is None:
        raise ValueError("seg is None")

    # Already a mask?
    if isinstance(seg, np.ndarray):
        m = seg
        if m.ndim == 3:
            m = m[..., 0]
        return m.astype(bool)

    # Sometimes seg comes as [rle_dict]
    if isinstance(seg, list) and len(seg) == 1 and isinstance(seg[0], dict):
        seg = seg[0]

    if not isinstance(seg, dict):
        raise TypeError(f"Unsupported seg type: {type(seg)}")

    if "size" not in seg or "counts" not in seg:
        raise ValueError(f"RLE dict missing size/counts. keys={list(seg.keys())}")

    size = seg["size"]
    # ---- Normalize size to (h,w) ----
    # Common cases: [H,W], [1,H,W], [H,W,1]
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        # Heuristic: take the last two as H,W when len==3
        if len(size) == 2:
            h, w = int(size[0]), int(size[1])
        else:
            h, w = int(size[-2]), int(size[-1])
    else:
        raise ValueError(f"Bad seg['size']={size}")

    counts = seg["counts"]

    # Convert torch/np -> python
    if torch.is_tensor(counts):
        counts = counts.detach().cpu().numpy()
    if isinstance(counts, np.ndarray):
        counts = counts.tolist()

    # If counts is nested like [[...]] flatten once
    if isinstance(counts, list) and len(counts) == 1 and isinstance(counts[0], list):
        counts = counts[0]

    # ---- Case A: uncompressed RLE list -> compress then decode ----
    if isinstance(counts, list):
        # Force int conversion (fixes your "int + str" crash)
        try:
            counts_int = [int(x) for x in counts]
        except Exception as e:
            # Print a small diagnostic and rethrow
            bad = next((x for x in counts if not str(x).lstrip("-").isdigit()), None)
            raise ValueError(f"Counts list contains non-integers. Example bad entry={bad!r}. Error={e}")

        rle_uc = {"size": [h, w], "counts": counts_int}
        rle = cocomask.frPyObjects(rle_uc, h, w)
        if isinstance(rle, list):
            rle = cocomask.merge(rle)

        m = cocomask.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        return m.astype(bool)

    # ---- Case B: compressed RLE string/bytes ----
    if isinstance(counts, str):
        rle = {"size": [h, w], "counts": counts.encode("utf-8")}
        m = cocomask.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        return m.astype(bool)

    if isinstance(counts, (bytes, bytearray)):
        rle = {"size": [h, w], "counts": bytes(counts)}
        m = cocomask.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        return m.astype(bool)

    raise TypeError(f"Unsupported counts type: {type(counts)}")

# =========================
# UDP sender (includes obj_name)
# =========================
class PoseUDPSender:
    def __init__(self, ip="127.0.0.1", port=5005):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, obj_name: str, t_m, q_wxyz, score=None, frame_id="camera_color_optical_frame"):
        payload = {
            "stamp": time.time(),
            "frame_id": frame_id,
            "obj_name": obj_name,
            "t_m": [float(t_m[0]), float(t_m[1]), float(t_m[2])],
            "q_wxyz": [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])],
        }
        if score is not None:
            payload["score"] = float(score)

        data = json.dumps(payload).encode("utf-8")
        self.sock.sendto(data, self.addr)


# =========================
# Multi-object tracker
# =========================
class MultiObjectTracker:
    def __init__(
        self,
        output_dir: str,
        objects: list,
        cam_K: list,
        depth_scale: float,
        visualize: bool = True,
        segmentor_model: str = "sam",
        stability_score_thresh: float = 0.97,
        det_score_thresh: float = 0.2,
        gpus: str = "0",
        pose_estimation_model: str = "pose_estimation_model",
        iter: int = 600000,
        exp_id: int = 0,
        assign_thresh: float = 0.20,
        topk_per_object: int = 3,
    ):
        """
        objects: list of dicts:
          { "name": str, "cad_path": str, "templates_dir": str }
        """
        self.output_dir = output_dir
        self.objects_cfg = objects
        self.cam_K = cam_K
        self.depth_scale = depth_scale
        self.visualize = visualize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.segmentor_model = segmentor_model
        self.stability_score_thresh = stability_score_thresh

        self.assign_thresh = float(assign_thresh)
        self.topk_per_object = int(topk_per_object)

        # Shared PEM cfg
        self.cfg = gorilla.Config.fromfile(osp.join(PEM_ROOT, "config", "base.yaml"))
        self.cfg.exp_name = pose_estimation_model + "_" + osp.splitext("Pose_Estimation_Model/config/base.yaml".split("/")[-1])[0] + "_id" + str(exp_id)
        self.cfg.gpus = gpus
        self.cfg.model_name = pose_estimation_model
        self.cfg.log_dir = osp.join("log", self.cfg.exp_name)
        self.cfg.test_iter = iter
        self.cfg.det_score_thresh = det_score_thresh

        gorilla.utils.set_cuda_visible_devices(gpu_ids=self.cfg.gpus)

        # Init shared ISM models
        self._init_shared_ism()

        # Init shared PEM net
        self._init_shared_pem_net()

        # Build per-object state
        self.objects = []
        for ocfg in self.objects_cfg:
            self.objects.append(self._build_object_state(ocfg["name"], ocfg["cad_path"], ocfg["templates_dir"]))

        logging.info(f"[MultiObjectTracker] Initialized {len(self.objects)} objects: {[o['name'] for o in self.objects]}")

    # ---------- Shared ISM ----------
    def _init_shared_ism(self):
        with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs"):
            cfg = compose(config_name="run_inference.yaml")

        if self.segmentor_model == "sam":
            with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
                cfg.model = compose(config_name="ISM_sam.yaml")
            cfg.model.segmentor_model.stability_score_thresh = self.stability_score_thresh
        elif self.segmentor_model == "fastsam":
            with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
                cfg.model = compose(config_name="ISM_fastsam.yaml")
        else:
            raise ValueError(f"segmentor_model={self.segmentor_model} not supported")

        if hasattr(cfg.model, "descriptor_model") and hasattr(cfg.model.descriptor_model, "checkpoint_dir"):
            cfg.model.descriptor_model.checkpoint_dir = os.path.join(CKPT_ROOT, "dinov2")

        logging.info("[ISM] Instantiating model...")
        self.model_segmentation = instantiate(cfg.model)

        # Move descriptor to device
        self.model_segmentation.descriptor_model.model = self.model_segmentation.descriptor_model.model.to(self.device)
        self.model_segmentation.descriptor_model.model.device = self.device

        # Move segmentor to device
        if hasattr(self.model_segmentation.segmentor_model, "predictor"):
            self.model_segmentation.segmentor_model.predictor.model = self.model_segmentation.segmentor_model.predictor.model.to(self.device)
        else:
            self.model_segmentation.segmentor_model.model.setup_model(device=self.device, verbose=True)

        logging.info(f"[ISM] Ready on {self.device}")

    # ---------- Shared PEM ----------
    def _init_shared_pem_net(self):
        random.seed(self.cfg.rd_seed)
        torch.manual_seed(self.cfg.rd_seed)

        logging.info("[PEM] Loading network...")
        MODEL = importlib.import_module(self.cfg.model_name)
        self.pose_estimation_model = MODEL.Net(self.cfg.model).cuda().eval()

        checkpoint = os.path.join(PEM_ROOT, "checkpoints", "sam-6d-pem-base.pth")
        gorilla.solver.load_checkpoint(model=self.pose_estimation_model, filename=checkpoint)
        logging.info("[PEM] Network loaded.")

    # ---------- Per-object state ----------
    def _compute_ism_ref(self, templates_dir: str):
        # expects templates_dir contains rgb_*.png and mask_*.png
        rgb_paths = sorted(glob.glob(osp.join(templates_dir, "rgb_*.png")))
        if len(rgb_paths) == 0:
            raise FileNotFoundError(f"No rgb_*.png in templates_dir={templates_dir}")

        boxes, masks, templates = [], [], []
        for p in rgb_paths:
            idx = osp.splitext(osp.basename(p))[0].split("_")[-1]
            image = Image.open(osp.join(templates_dir, f"rgb_{idx}.png"))
            mask = Image.open(osp.join(templates_dir, f"mask_{idx}.png"))

            boxes.append(mask.getbbox())

            image_t = torch.from_numpy(np.array(image.convert("RGB")) / 255.0).float()
            mask_t = torch.from_numpy(np.array(mask.convert("L")) / 255.0).float()
            image_t = image_t * mask_t[:, :, None]
            templates.append(image_t)
            masks.append(mask_t.unsqueeze(-1))

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))

        proposal_processor = CropResizePad(224)
        templates = proposal_processor(images=templates, boxes=boxes).to(self.device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(self.device)

        ref_desc = self.model_segmentation.descriptor_model.compute_features(
            templates, token_name="x_norm_clstoken"
        ).unsqueeze(0).data

        ref_appe = self.model_segmentation.descriptor_model.compute_masked_patch_feature(
            templates, masks_cropped[:, 0, :, :]
        ).unsqueeze(0).data

        return ref_desc, ref_appe

    def _compute_geom_ref(self, cad_path: str):
        mesh = trimesh.load_mesh(cad_path)
        model_points = mesh.sample(2048).astype(np.float32) / 1000.0
        ref_pc = torch.tensor(model_points).unsqueeze(0).to(self.device)

        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
        ref_poses = poses[load_index_level_in_level2(0, "all"), :, :]
        return ref_pc, ref_poses

    def _compute_pem_ref(self, cad_path: str, templates_dir: str):
        # model points + radius
        mesh = trimesh.load_mesh(cad_path)
        model_points_pem = mesh.sample(self.cfg.test_dataset.n_sample_model_point).astype(np.float32) / 1000.0
        radius_pem = float(np.max(np.linalg.norm(model_points_pem, axis=1)))

        # template extraction (same logic you had)
        def _get_template(path, cfg, tem_index=1):
            rgb_path = os.path.join(path, f"rgb_{tem_index}.png")
            mask_path = os.path.join(path, f"mask_{tem_index}.png")
            xyz_path = os.path.join(path, f"xyz_{tem_index}.npy")

            rgb = load_im(rgb_path).astype(np.uint8)
            xyz = np.load(xyz_path).astype(np.float32) / 1000.0
            mask = load_im(mask_path).astype(np.uint8) == 255

            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
            mask_c = mask[y1:y2, x1:x2]

            rgb_c = rgb[:, :, ::-1][y1:y2, x1:x2, :]
            if cfg.rgb_mask_flag:
                rgb_c = rgb_c * (mask_c[:, :, None] > 0).astype(np.uint8)

            rgb_c = cv2.resize(rgb_c, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
            rgb_c = rgb_transform(np.array(rgb_c))

            choose = (mask_c > 0).astype(np.float32).flatten().nonzero()[0]
            if len(choose) <= cfg.n_sample_template_point:
                choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
            choose = choose[choose_idx]
            xyz_c = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
            return rgb_c, rgb_choose, xyz_c

        def get_templates(path, cfg):
            n_template_view = cfg.n_template_view
            all_tem, all_tem_choose, all_tem_pts = [], [], []

            total_nView = 42
            for v in range(n_template_view):
                i = int(total_nView / n_template_view * v)
                tem, tem_choose, tem_pts = _get_template(path, cfg, i)
                all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
                all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
                all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
            return all_tem, all_tem_pts, all_tem_choose

        all_tem, all_tem_pts, all_tem_choose = get_templates(templates_dir, self.cfg.test_dataset)
        with torch.no_grad():
            all_tem_pts_t, all_tem_feat_t = self.pose_estimation_model.feature_extraction.get_obj_feats(
                all_tem, all_tem_pts, all_tem_choose
            )

        return model_points_pem, radius_pem, all_tem_pts_t, all_tem_feat_t

    def _build_object_state(self, name: str, cad_path: str, templates_dir: str):
        if not osp.exists(cad_path):
            raise FileNotFoundError(f"[{name}] cad_path not found: {cad_path}")
        if not osp.isdir(templates_dir):
            raise FileNotFoundError(f"[{name}] templates_dir not found: {templates_dir}")

        logging.info(f"[{name}] Building ISM refs from templates: {templates_dir}")
        ref_desc, ref_appe = self._compute_ism_ref(templates_dir)

        logging.info(f"[{name}] Building geometric refs from CAD: {cad_path}")
        ref_pc, ref_poses = self._compute_geom_ref(cad_path)

        logging.info(f"[{name}] Building PEM refs (template feats + model pts)")
        model_points_pem, radius_pem, all_tem_pts, all_tem_feat = self._compute_pem_ref(cad_path, templates_dir)

        return {
            "name": name,
            "cad_path": cad_path,
            "templates_dir": templates_dir,
            # ISM
            "ref_desc": ref_desc,
            "ref_appe": ref_appe,
            "ref_pc": ref_pc,
            "ref_poses": ref_poses,
            # PEM
            "model_points_pem": model_points_pem,
            "radius_pem": radius_pem,
            "all_tem_pts": all_tem_pts,
            "all_tem_feat": all_tem_feat,
        }
    
    def get_best_segmentation_detections(self, dets_by_obj: dict):
        """ Return one best segmentation detection per object """
        best_dets = []
        for obj in self.objects:
            name = obj["name"]
            dets = dets_by_obj.get(name, [])
            if not dets:
                continue
            best_det = max(dets, key=lambda d: float(d.get("score", -1e9)))
            best_dets.append(best_det)
        return best_dets

    # =========================
    # Visualization (ISM)
    # =========================
    def visualize_ism(self, rgb: Image.Image, detections, message: str):
        left = rgb.convert("RGB")
        left_np = np.array(left)

        gray = cv2.cvtColor(left_np, cv2.COLOR_RGB2GRAY)
        right_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if message is not None:
            cv2.putText(right_np, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)
            right = Image.fromarray(right_np.astype(np.uint8))
        else:
            colors = distinctipy.get_colors(max(1, len(detections)))
            alpha = 0.33

            best_det = max(detections, key=lambda d: d.get("score", 0.0))
            mask = decode_coco_segmentation(best_det["segmentation"])

            edge = canny(mask)
            edge = binary_dilation(edge, np.ones((2, 2)))

            obj_id = int(best_det.get("category_id", 1))
            temp_id = max(obj_id - 1, 0)
            temp_id = min(temp_id, len(colors) - 1)

            r = int(255 * colors[temp_id][0])
            g = int(255 * colors[temp_id][1])
            b = int(255 * colors[temp_id][2])

            right_np[mask, 0] = alpha * r + (1 - alpha) * right_np[mask, 0]
            right_np[mask, 1] = alpha * g + (1 - alpha) * right_np[mask, 1]
            right_np[mask, 2] = alpha * b + (1 - alpha) * right_np[mask, 2]
            right_np[edge, :] = 255

            right = Image.fromarray(np.uint8(right_np))

        concat = Image.new("RGB", (left.width + right.width, left.height))
        concat.paste(left, (0, 0))
        concat.paste(right, (left.width, 0))
        return concat

    # =========================
    # Visualization (PEM)
    # =========================
    def visualize_pem(self, rgb_bgr: np.ndarray, message: str = None,
                      pred_rot=None, pred_trans=None, model_points=None, K=None):
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        left = Image.fromarray(rgb.astype(np.uint8))

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        right_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if message is not None:
            cv2.putText(right_np, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)
            right = Image.fromarray(right_np.astype(np.uint8))
        else:
            if pred_rot is None or pred_trans is None or model_points is None or K is None:
                cv2.putText(right_np, "Pose args missing", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2, cv2.LINE_AA)
                right = Image.fromarray(right_np.astype(np.uint8))
            else:
                overlay = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
                right = Image.fromarray(overlay.astype(np.uint8))

        concat = Image.new("RGB", (left.width + right.width, left.height))
        concat.paste(left, (0, 0))
        concat.paste(right, (left.width, 0))
        return concat

    def visualize_pem_best_per_object(self, color_bgr: np.ndarray, poses_by_obj: dict):
        """ Render one best pose per object on a single shared overlay """
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        left = Image.fromarray(rgb.astype(np.uint8))

        overlay = rgb.copy()
        K = np.array(self.cam_K).reshape(3, 3)

        drew_any = False

        for obj in self.objects:
            name = obj["name"]
            dets = poses_by_obj.get(name, [])
            if not dets:
                continue

            best_det = max(dets, key=lambda d: float(d.get("score", -1e9)))
            if "R" not in best_det or "t" not in best_det:
                continue

            R = np.array(best_det["R"], dtype=np.float32)
            t = np.array(best_det["t"], dtype=np.float32)

            if R.shape != (3, 3) or t.shape != (3,):
                continue

            overlay = draw_detections(
                overlay,
                R[None, ...],
                t[None, ...],
                obj["model_points_pem"] * 1000.0,
                K[None, ...],
                color=(255, 0, 0),
            )
            drew_any = True

        if not drew_any:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            right_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cv2.putText(
                right_np, "PEM: no poses", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
            )
            right = Image.fromarray(right_np.astype(np.uint8))
        else:
            right = Image.fromarray(overlay.astype(np.uint8))

        concat = Image.new("RGB", (left.width + right.width, left.height))
        concat.paste(left, (0, 0))
        concat.paste(right, (left.width, 0))
        return concat

    # =========================
    # Multi-object ISM inference:
    # - generate masks once
    # - compute scores for each object
    # - assign proposals to best object
    # returns: (vis_ism, dets_by_obj_name)
    # =========================
    def run_segmentation_multi(self, color_bgr: np.ndarray, depth_bop: np.ndarray):
        whole_image = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rgb_pil = Image.fromarray(whole_image) if self.visualize else None

        masks = self.model_segmentation.segmentor_model.generate_masks(whole_image)
        if masks is None or len(masks) == 0:
            if self.visualize:
                return self.visualize_ism(rgb_pil, None, "No masks detected"), {}
            return None, {}

        detections = Detections(masks)

        # Descriptor forward (once)
        try:
            query_desc, query_appe = self.model_segmentation.descriptor_model.forward(whole_image, detections)
        except Exception as e:
            msg = f"Descriptor failed: {e}"
            logging.warning(msg)
            if self.visualize:
                return self.visualize_ism(rgb_pil, None, msg), {}
            return None, {}

        # batch for geometry
        depth = np.array(depth_bop).astype(np.int32)
        cam_K = np.array(self.cam_K).reshape((3, 3))
        depth_scale = np.array(self.depth_scale)
        batch = {
            "depth": torch.from_numpy(depth).unsqueeze(0).to(self.device),
            "cam_intrinsic": torch.from_numpy(cam_K).unsqueeze(0).to(self.device),
            "depth_scale": torch.from_numpy(depth_scale).unsqueeze(0).to(self.device),
        }

        # Ensure masks tensor dims are stable
        if hasattr(detections, "masks") and detections.masks is not None:
            m = detections.masks
            if hasattr(m, "dim") and m.dim() == 2:
                detections.masks = m.unsqueeze(0)
            elif isinstance(m, np.ndarray) and m.ndim == 2:
                detections.masks = m[None, ...]

        n_prop = len(detections)
        n_obj = len(self.objects)
        score_mat = np.full((n_obj, n_prop), -1e9, dtype=np.float32)

        # compute total scores per object (re-using existing ISM scoring functions)
        for oi, obj in enumerate(self.objects):
            # Swap in per-object ref_data
            self.model_segmentation.ref_data = {
                "descriptors": obj["ref_desc"],
                "appe_descriptors": obj["ref_appe"],
                "pointcloud": obj["ref_pc"],
                "poses": obj["ref_poses"],
            }

            try:
                idx_sel, pred_idx_obj, semantic_score, best_template = self.model_segmentation.compute_semantic_score(query_desc)
            except Exception as e:
                logging.warning(f"[{obj['name']}] compute_semantic_score failed: {e}")
                continue

            if idx_sel is None or len(idx_sel) == 0:
                continue

            # appearance
            try:
                appe_scores, ref_aux = self.model_segmentation.compute_appearance_score(
                    best_template, pred_idx_obj, query_appe[idx_sel, :]
                )
            except Exception as e:
                logging.warning(f"[{obj['name']}] compute_appearance_score failed: {e}")
                continue

            # geometry (best-effort; fallback to 0 if fails)
            try:
                # NOTE: project_template_to_image expects masks aligned with detections; we pass subset
                image_uv = self.model_segmentation.project_template_to_image(
                    best_template, pred_idx_obj, batch, detections.masks[idx_sel]
                )
                geo_score, vis_ratio = self.model_segmentation.compute_geometric_score(
                    image_uv, detections, query_appe[idx_sel, :], ref_aux,
                    visible_thred=self.model_segmentation.visible_thred
                )
            except Exception as e:
                logging.debug(f"[{obj['name']}] geometry skipped: {e}")
                geo_score = torch.zeros_like(appe_scores)
                vis_ratio = torch.ones_like(appe_scores)

            # total score per selected proposal
            # semantic_score is typically per-proposal (same indexing as detections), but some impls return only selected.
            # To be safe, gather if needed:
            if hasattr(semantic_score, "__len__") and len(semantic_score) == n_prop:
                sem_sel = semantic_score[idx_sel]
            else:
                sem_sel = semantic_score

            total = (sem_sel + appe_scores + geo_score * vis_ratio) / (1.0 + 1.0 + vis_ratio)
            total_np = total.detach().cpu().numpy().astype(np.float32)

            # write into full proposal axis
            for k, pidx in enumerate(idx_sel):
                pidx_i = int(pidx.item()) if hasattr(pidx, "item") else int(pidx)
                score_mat[oi, pidx_i] = total_np[k]

        best_obj = score_mat.argmax(axis=0)
        best_score = score_mat.max(axis=0)
        valid = best_score > self.assign_thresh

        # Ensure attrs exist even if upstream code expects them
        detections.add_attribute("scores", torch.from_numpy(best_score).to(self.device).to(torch.float32))
        detections.add_attribute("object_ids", torch.from_numpy(best_obj).to(self.device).to(torch.long))

        # Convert detections to JSON once; pick by proposal index
        det_json_all = detections.convert_to_json(scene_id=0, image_id=0, runtime=0, dataset_name="Custom")

        dets_by_obj = {}
        for oi, obj in enumerate(self.objects):
            keep_idx = np.where(valid & (best_obj == oi))[0]
            obj_dets = []
            for pidx in keep_idx:
                d = det_json_all[int(pidx)]
                d["score"] = float(best_score[int(pidx)])
                d["category_id"] = int(oi + 1)
                obj_dets.append(d)
            # Sort high->low and keep topK for PEM efficiency
            obj_dets.sort(key=lambda x: float(x.get("score", -1e9)), reverse=True)
            dets_by_obj[obj["name"]] = obj_dets[: self.topk_per_object]

        # ISM visualization: overlay best valid overall, else message
        vis = None
        if self.visualize:
            if valid.any():
                pidx = int(np.argmax(best_score))
                # show that detection only for readability
                vis = self.visualize_ism(rgb_pil, [det_json_all[pidx]], message=None)
            else:
                vis = self.visualize_ism(rgb_pil, None, "No assigned detections")
        return vis, dets_by_obj

    # =========================
    # PEM per-object inference
    # returns (vis_pem, dets_with_pose)
    # =========================
    def run_pose_estimation_for_object(self, obj, color_bgr: np.ndarray, depth_bop: np.ndarray, detections_json: list):
        whole_image = color_bgr.astype(np.uint8)

        if depth_bop is None:
            if self.visualize:
                return self.visualize_pem(whole_image, message=f"{obj['name']} PEM: no depth"), []
            return None, []

        if detections_json is None or len(detections_json) == 0:
            if self.visualize:
                return self.visualize_pem(whole_image, message=f"{obj['name']} PEM: no dets"), []
            return None, []

        K = np.array(self.cam_K).reshape(3, 3)

        # Filter by thresh
        dets = [d for d in detections_json if float(d.get("score", 0.0)) > float(self.cfg.det_score_thresh)]
        if len(dets) == 0:
            if self.visualize:
                return self.visualize_pem(whole_image, message=f"{obj['name']} PEM: det<th"), []
            return None, []

        # Prepare depth/point cloud
        if len(whole_image.shape) == 2:
            whole_image = np.concatenate([whole_image[:, :, None]] * 3, axis=2)

        whole_depth = depth_bop.astype(np.float32) * self.depth_scale / 1000.0
        if np.count_nonzero(whole_depth) == 0:
            if self.visualize:
                return self.visualize_pem(whole_image, message=f"{obj['name']} PEM: empty depth"), []
            return None, []

        whole_pts = get_point_cloud_from_depth(whole_depth, K)

        all_rgb, all_cloud, all_rgb_choose, all_score, all_dets = [], [], [], [], []

        for inst in dets:
            seg = inst["segmentation"]
            score = float(inst.get("score", 0.0))

            mask = decode_coco_segmentation(seg)
            mask = np.logical_and(mask, whole_depth > 0)

            if np.sum(mask) <= 32:
                continue

            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox

            mask_c = mask[y1:y2, x1:x2]
            choose = mask_c.astype(np.float32).flatten().nonzero()[0]

            cloud = whole_pts[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
            if cloud.shape[0] < 8:
                continue

            center = np.mean(cloud, axis=0)
            tmp_cloud = cloud - center[None, :]
            flag = np.linalg.norm(tmp_cloud, axis=1) < obj["radius_pem"] * 1.2

            if np.sum(flag) < 4:
                continue

            choose = choose[flag]
            cloud = cloud[flag]

            # sample points
            n_obs = int(self.cfg.test_dataset.n_sample_observed_point)
            if len(choose) <= n_obs:
                choose_idx = np.random.choice(np.arange(len(choose)), n_obs)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), n_obs, replace=False)

            choose = choose[choose_idx]
            cloud = cloud[choose_idx]

            # rgb crop
            rgb = whole_image[y1:y2, x1:x2, :][:, :, ::-1]
            if self.cfg.test_dataset.rgb_mask_flag:
                rgb = rgb * (mask_c[:, :, None] > 0).astype(np.uint8)

            rgb = cv2.resize(rgb, (self.cfg.test_dataset.img_size, self.cfg.test_dataset.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = rgb_transform(np.array(rgb))

            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.cfg.test_dataset.img_size)

            all_rgb.append(torch.FloatTensor(rgb))
            all_cloud.append(torch.FloatTensor(cloud))
            all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
            all_score.append(score)
            all_dets.append(inst)

        if len(all_dets) == 0:
            if self.visualize:
                return self.visualize_pem(whole_image, message=f"{obj['name']} PEM: no valid crops"), []
            return None, []

        ret_dict = {
            "pts": torch.stack(all_cloud).cuda(),
            "rgb": torch.stack(all_rgb).cuda(),
            "rgb_choose": torch.stack(all_rgb_choose).cuda(),
            "score": torch.FloatTensor(all_score).cuda(),
        }

        ninstance = ret_dict["pts"].size(0)
        ret_dict["model"] = torch.FloatTensor(obj["model_points_pem"]).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        ret_dict["K"] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()

        with torch.no_grad():
            ret_dict["dense_po"] = obj["all_tem_pts"].repeat(ninstance, 1, 1)
            ret_dict["dense_fo"] = obj["all_tem_feat"].repeat(ninstance, 1, 1)
            out = self.pose_estimation_model(ret_dict)

        if "pred_pose_score" in out.keys():
            pose_scores = out["pred_pose_score"] * out["score"]
        else:
            pose_scores = out["score"]

        pose_scores = pose_scores.detach().cpu().numpy()
        pred_rot = out["pred_R"].detach().cpu().numpy()
        pred_trans = out["pred_t"].detach().cpu().numpy() * 1000.0  # mm

        for idx, det in enumerate(all_dets):
            det["score"] = float(pose_scores[idx])
            det["R"] = list(pred_rot[idx].tolist())
            det["t"] = list(pred_trans[idx].tolist())

        # choose best instance for vis
        best_idx = int(np.argmax(pose_scores))
        valid_mask = np.zeros_like(pose_scores, dtype=bool)
        valid_mask[best_idx] = True

        if self.visualize:
            K_vis = ret_dict["K"].detach().cpu().numpy()[valid_mask]
            vis_img = self.visualize_pem(
                whole_image,
                message=None,
                pred_rot=pred_rot[valid_mask],
                pred_trans=pred_trans[valid_mask],
                model_points=obj["model_points_pem"] * 1000.0,
                K=K_vis,
            )
            return vis_img, all_dets

        return None, all_dets


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Live SAM-6D MULTI-object inference from RealSense stream.")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory.")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization (imshow).")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="Disable visualization for max FPS.")
    parser.set_defaults(visualize=True)

    # Multi-object args: repeatable
    parser.add_argument("--obj_name", action="append", default=[], help="Repeat: object name.")
    parser.add_argument("--cad_path", action="append", default=[], help="Repeat: CAD path (ply) for the object.")
    parser.add_argument("--templates_dir", action="append", default=[], help="Repeat: templates dir for object. Default: <output_dir>/<obj_name>/templates")

    # Segmentor
    parser.add_argument("--segmentor_model", default="sam", choices=["sam", "fastsam"])
    parser.add_argument("--stability_score_thresh", default=0.97, type=float)

    # PEM
    parser.add_argument("--det_score_thresh", default=0.2, type=float)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--pose_estimation_model", type=str, default="pose_estimation_model")
    parser.add_argument("--iter", type=int, default=600000)
    parser.add_argument("--exp_id", type=int, default=0)

    # Multi-object assignment + PEM speed
    parser.add_argument("--assign_thresh", type=float, default=0.20, help="Min ISM score to assign a proposal to an object.")
    parser.add_argument("--topk_per_object", type=int, default=3, help="Keep top-K ISM detections per object for PEM.")

    # UDP
    parser.add_argument("--udp_ip", type=str, default="127.0.0.1")
    parser.add_argument("--udp_port", type=int, default=5005)

    # RealSense
    parser.add_argument("--realsense_serial", type=str, default="036322250488")

    args = parser.parse_args()

    if len(args.cad_path) == 0:
        raise ValueError("Provide at least one --cad_path (repeatable).")

    if len(args.obj_name) not in (0, len(args.cad_path)):
        raise ValueError("--obj_name must be omitted or match number of --cad_path.")

    if len(args.obj_name) == 0:
        args.obj_name = [osp.splitext(osp.basename(p))[0] for p in args.cad_path]

    if len(args.templates_dir) not in (0, len(args.cad_path)):
        raise ValueError("--templates_dir must be omitted or match number of objects.")

    if len(args.templates_dir) == 0:
        args.templates_dir = [osp.join(args.output_dir, n, "outputs/templates") for n in args.obj_name]

    # Build objects list
    objects = []
    for n, cad, tdir in zip(args.obj_name, args.cad_path, args.templates_dir):
        objects.append({"name": n, "cad_path": cad, "templates_dir": tdir})

    # UDP sender
    udp = PoseUDPSender(ip=args.udp_ip, port=args.udp_port)

    # RealSense
    realsense = RealSenseCamera(
        serial_number=args.realsense_serial,
        depth_scale=1.0,
        intrinsics_for="color",
        out_dir=args.output_dir,
        align_to_color=True,
    )
    camera_intrinsics = realsense.get_camera_intrinsics(save_json=True, print_info=True)
    print("Camera intrinsics BOP:\n", camera_intrinsics["bop"])

    tracker = MultiObjectTracker(
        output_dir=args.output_dir,
        objects=objects,
        cam_K=camera_intrinsics["bop"]["cam_K"],
        depth_scale=camera_intrinsics["bop"]["depth_scale"],
        visualize=args.visualize,
        segmentor_model=args.segmentor_model,
        stability_score_thresh=args.stability_score_thresh,
        det_score_thresh=args.det_score_thresh,
        gpus=args.gpus,
        pose_estimation_model=args.pose_estimation_model,
        iter=args.iter,
        exp_id=args.exp_id,
        assign_thresh=args.assign_thresh,
        topk_per_object=args.topk_per_object,
    )

    if args.visualize:
        window_name = "SAM-6D Multi-object (q=quit, s=save)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 800)   

    try:
        frame_i = 0
        for color_bgr, depth_bop in realsense.frames():
            frame_i += 1

            vis_img_ism, dets_by_obj = tracker.run_segmentation_multi(color_bgr, depth_bop)

            # Run PEM per object
            pem_vis_panels = []
            poses_by_obj = {}

            for obj in tracker.objects:
                name = obj["name"]
                vis_pem, dets_pose = tracker.run_pose_estimation_for_object(obj, color_bgr, depth_bop, dets_by_obj.get(name, []))
                poses_by_obj[name] = dets_pose

                # UDP send best per object
                if dets_pose:
                    best_det = max(dets_pose, key=lambda d: float(d.get("score", -1e9)))
                    if "R" in best_det and "t" in best_det:
                        R_best = np.squeeze(np.array(best_det["R"], dtype=np.float32))
                        t_best = np.squeeze(np.array(best_det["t"], dtype=np.float32))
                        if R_best.shape == (3, 3) and t_best.shape == (3,):
                            q_best = rotmat_to_quat_wxyz(R_best)
                            t_m = t_best / 1000.0
                            if np.isfinite(t_m).all() and np.isfinite(q_best).all():
                                udp.send(name, t_m, q_best, score=float(best_det.get("score", 0.0)))
                                print(f"[{name}] score={float(best_det.get('score', 0.0)):.4f} "
                                      f"t(m)=[{t_m[0]:.3f}, {t_m[1]:.3f}, {t_m[2]:.3f}] "
                                      f"q(wxyz)=[{q_best[0]:.5f}, {q_best[1]:.5f}, {q_best[2]:.5f}, {q_best[3]:.5f}]")

                if args.visualize and vis_pem is not None:
                    pem_vis_panels.append(np.array(vis_pem))

            # Visualization panel: ISM on top + PEM panels below
            if args.visualize:
                rgb_pil = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))

                # one best segmentation per object
                best_seg_dets = tracker.get_best_segmentation_detections(dets_by_obj)
                if len(best_seg_dets) > 0:
                    vis_img_ism = tracker.visualize_ism(rgb_pil, best_seg_dets, message=None)
                else:
                    vis_img_ism = tracker.visualize_ism(rgb_pil, None, message="No assigned detections")

                # one best pose per object
                vis_img_pem = tracker.visualize_pem_best_per_object(color_bgr, poses_by_obj)

                vis_ism_np = np.array(vis_img_ism)
                vis_pem_np = np.array(vis_img_pem)

                # make widths match
                target_w = max(vis_ism_np.shape[1], vis_pem_np.shape[1])

                if vis_ism_np.shape[1] != target_w:
                    scale = target_w / float(vis_ism_np.shape[1])
                    vis_ism_np = cv2.resize(vis_ism_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                if vis_pem_np.shape[1] != target_w:
                    scale = target_w / float(vis_pem_np.shape[1])
                    vis_pem_np = cv2.resize(vis_pem_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                vis_stack = np.vstack([vis_ism_np, vis_pem_np])

                vis_bgr = cv2.cvtColor(vis_stack, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, vis_bgr)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("s"):
                    out_dir = os.path.join(args.output_dir, "sam6d_results_multi")
                    os.makedirs(out_dir, exist_ok=True)

                    with open(os.path.join(out_dir, f"detections_ism_by_obj_frame{frame_i}.json"), "w") as f:
                        json.dump(dets_by_obj, f, indent=2)

                    with open(os.path.join(out_dir, f"detections_pem_by_obj_frame{frame_i}.json"), "w") as f:
                        json.dump(poses_by_obj, f, indent=2)

                    Image.fromarray(vis_stack.astype(np.uint8)).save(
                        os.path.join(out_dir, f"vis_multi_frame{frame_i}.png")
                    )
                    print(f"Saved results to {out_dir}")
                    
    finally:
        if args.visualize:
            cv2.destroyAllWindows()
        del realsense


if __name__ == "__main__":
    main()
