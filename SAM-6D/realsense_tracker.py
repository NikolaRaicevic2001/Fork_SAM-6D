#!/usr/bin/env python3
import os
import cv2
import sys
import json
import glob
import torch
import random
import gorilla
import trimesh
import imageio
import logging
import argparse
import importlib
import distinctipy
import numpy as np
import os.path as osp
import pycocotools.mask as cocomask

from tqdm import tqdm
from PIL import Image
from skimage.feature import canny
from hydra.utils import instantiate
from hydra import initialize, compose
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

# Define repository paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISM_ROOT = os.path.join(BASE_DIR, "Instance_Segmentation_Model")
PEM_ROOT = os.path.join(BASE_DIR, "Pose_Estimation_Model")
CKPT_ROOT = os.path.join(ISM_ROOT, "checkpoints")

if ISM_ROOT not in sys.path:
    sys.path.insert(0, ISM_ROOT)
sys.path.append(os.path.join(PEM_ROOT, 'provider'))
sys.path.append(os.path.join(PEM_ROOT, 'utils'))
sys.path.append(os.path.join(PEM_ROOT, 'model'))
sys.path.append(os.path.join(PEM_ROOT, 'model', 'pointnet2'))

from camera import RealSenseCamera
from draw_utils import draw_detections
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad
from Instance_Segmentation_Model.utils.inout import load_json, save_json_bop23
from Instance_Segmentation_Model.model.utils import Detections, convert_npz_to_json
from data_utils import ( load_im, get_bbox, get_point_cloud_from_depth, get_resize_rgb_choose)
from Instance_Segmentation_Model.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2


logging.basicConfig(level=logging.INFO)

# Hyperparameters
rgb_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# ObjectTracker
class ObjectTracker:
    def __init__( self, 
                  output_dir: str,
                  cad_path: str,
                  cam_K: list,
                  depth_scale: float,
                  segmentor_model: str = "sam",
                  stability_score_thresh: float = 0.97,
                  det_score_thresh: float = 0.2,
                  gpus: str = "0",
                  pose_estimation_model: str = "pose_estimation_model",
                  iter: int = 600000,
                  exp_id: int = 0,
                ):
        """ Initialize the Object Tracker with segmentation and descriptor models """
        # Initialize parameters
        self.output_dir = output_dir
        self.cad_path = cad_path
        self.cam_K = cam_K
        self.depth_scale = depth_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Segmentation parameters
        self.segmentor_model = segmentor_model
        self.stability_score_thresh = stability_score_thresh

        # Initialize Pose Estimation parameters
        self.cfg = gorilla.Config.fromfile(osp.join(PEM_ROOT, "config", "base.yaml"))
        self.cfg.exp_name = pose_estimation_model + '_' + osp.splitext("Pose_Estimation_Model/config/base.yaml".split("/")[-1])[0] + '_id' + str(exp_id)
        self.cfg.gpus     = gpus
        self.cfg.model_name = pose_estimation_model
        self.cfg.log_dir  = osp.join("log", self.cfg.exp_name)
        self.cfg.test_iter = iter
        self.cfg.det_score_thresh = det_score_thresh
        gorilla.utils.set_cuda_visible_devices(gpu_ids = self.cfg.gpus)

        # # Initialize Segmentation Model
        # self.initialize_segmentation_model()

        # Initialize Pose Estimation Model
        self.initialize_pose_estimation_model()

    def initialize_segmentation_model(self):
        """ Initialize segmentation model """
        # Initialize Segmentation Model Configuration
        with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs"):
            cfg = compose(config_name='run_inference.yaml')
        if self.segmentor_model == "sam":
            with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
                cfg.model = compose(config_name='ISM_sam.yaml')
            cfg.model.segmentor_model.stability_score_thresh = self.stability_score_thresh
        elif self.segmentor_model == "fastsam":
            with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
                cfg.model = compose(config_name='ISM_fastsam.yaml')
        else:
            raise ValueError("The segmentor_model {} is not supported now!".format(self.segmentor_model))
        
        if hasattr(cfg.model, "descriptor_model") and hasattr(cfg.model.descriptor_model, "checkpoint_dir"):
            print("Using DINOv2 Vit-L14 pretrained model as descriptor model directory.")
            cfg.model.descriptor_model.checkpoint_dir = os.path.join(CKPT_ROOT, "dinov2")

        logging.info("Initializing model")
        self.model_segmentation = instantiate(cfg.model)
        self.model_segmentation.descriptor_model.model = self.model_segmentation.descriptor_model.model.to(self.device)
        self.model_segmentation.descriptor_model.model.device = self.device

        # if there is predictor in the model, move it to device
        if hasattr(self.model_segmentation.segmentor_model, "predictor"):
            self.model_segmentation.segmentor_model.predictor.model = (self.model_segmentation.segmentor_model.predictor.model.to(self.device))
        else:
            self.model_segmentation.segmentor_model.model.setup_model(device=self.device, verbose=True)
        logging.info(f"Moving models to {self.device} done!")
    
        logging.info("Initializing template")
        template_dir = os.path.join(self.output_dir, 'templates')
        num_templates = len(glob.glob(f"{template_dir}/*.npy"))
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
            mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
            boxes.append(mask.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks.append(mask.unsqueeze(-1))
            
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        processing_config = OmegaConf.create({"image_size": 224,})
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).to(self.device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(self.device)

        self.model_segmentation.ref_data = {}
        self.model_segmentation.ref_data["descriptors"] = self.model_segmentation.descriptor_model.compute_features(templates, token_name="x_norm_clstoken").unsqueeze(0).data
        self.model_segmentation.ref_data["appe_descriptors"] = self.model_segmentation.descriptor_model.compute_masked_patch_feature(templates, masks_cropped[:, 0, :, :]).unsqueeze(0).data
        

    def initialize_pose_estimation_model(self):
        """ Initialize pose estimation model """
        def _get_template(path, cfg, tem_index=1):
            rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
            mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
            xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

            rgb = load_im(rgb_path).astype(np.uint8)
            xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
            mask = load_im(mask_path).astype(np.uint8) == 255

            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
            mask = mask[y1:y2, x1:x2]

            rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
            if cfg.rgb_mask_flag:
                rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

            rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = rgb_transform(np.array(rgb))

            choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
            if len(choose) <= cfg.n_sample_template_point:
                choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
            choose = choose[choose_idx]
            xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
            return rgb, rgb_choose, xyz

        def get_templates(path, cfg):
            n_template_view = cfg.n_template_view
            all_tem = []
            all_tem_choose = []
            all_tem_pts = []

            total_nView = 42
            for v in range(n_template_view):
                i = int(total_nView / n_template_view * v)
                tem, tem_choose, tem_pts = _get_template(path, cfg, i)
                all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
                all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
                all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
            return all_tem, all_tem_pts, all_tem_choose
    
        random.seed(self.cfg.rd_seed)
        torch.manual_seed(self.cfg.rd_seed)

        # model
        print("=> creating model ...")
        MODEL = importlib.import_module(self.cfg.model_name)
        self.pose_estimation_model = MODEL.Net(self.cfg.model)
        self.pose_estimation_model = self.pose_estimation_model.cuda()
        self.pose_estimation_model.eval()
        checkpoint = os.path.join(PEM_ROOT, 'checkpoints', 'sam-6d-pem-base.pth')
        gorilla.solver.load_checkpoint(model=self.pose_estimation_model, filename=checkpoint)
        print("=> extracting templates ...")
        tem_path = os.path.join(self.output_dir, 'templates')
        all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, self.cfg.test_dataset)
        with torch.no_grad():
            self.all_tem_pts, self.all_tem_feat = self.pose_estimation_model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)


    def batch_input_data(self, depth_bop: np.array) -> dict:
        """ Prepare batch input data from depth image """
        batch = {}
        depth = np.array(depth_bop).astype(np.int32)
        cam_K = np.array(self.cam_K).reshape((3, 3))
        depth_scale = np.array(self.depth_scale)

        batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(self.device)
        batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(self.device)
        batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(self.device)
        return batch

    def visualize(self, rgb: Image.Image, detections, save_path=None):
        img = rgb.copy()
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        colors = distinctipy.get_colors(len(detections))
        alpha = 0.33

        best_score = -1e9
        best_det = None
        for det in detections:
            if det.get("score", 0.0) > best_score:
                best_score = det.get("score", 0.0)
                best_det = det

        if best_det is None:
            return self.visualize_fallback(rgb, "No best det")

        mask = rle_to_mask(best_det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))

        obj_id = int(best_det["category_id"])
        temp_id = max(obj_id - 1, 0)

        r = int(255 * colors[temp_id][0])
        g = int(255 * colors[temp_id][1])
        b = int(255 * colors[temp_id][2])

        img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]
        img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]
        img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]
        img[edge, :] = 255

        prediction = Image.fromarray(np.uint8(img))

        # side-by-side
        concat = Image.new("RGB", (rgb.width + prediction.width, rgb.height))
        concat.paste(rgb.convert("RGB"), (0, 0))
        concat.paste(prediction, (rgb.width, 0))

        # If caller explicitly wants to save
        if save_path is not None:
            concat.save(save_path)

        return concat

    def visualize_fallback(self, rgb: Image.Image, message: str = "No detection"):
        """ Returns a PIL.Image with the same layout as visualize() with a message """
        left = rgb.convert("RGB")
        left_np = np.array(left)

        gray = cv2.cvtColor(left_np, cv2.COLOR_RGB2GRAY)
        right_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Text overlay
        cv2.putText(
            right_np,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        right = Image.fromarray(right_np)
        concat = Image.new("RGB", (left.width + right.width, left.height))
        concat.paste(left, (0, 0))
        concat.paste(right, (left.width, 0))
        return concat
    
    def visualize_pose_estimation(self, rgb, pred_rot, pred_trans, model_points, K, save_path):
        img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
        img = Image.fromarray(np.uint8(img))
        img.save(save_path)
        prediction = Image.open(save_path)
        
        # concat side by side in PIL
        rgb = Image.fromarray(np.uint8(rgb))
        img = np.array(img)
        concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
        concat.paste(rgb, (0, 0))
        concat.paste(prediction, (img.shape[1], 0))
        return concat

    def run_segmentation_inference(self, color_bgr: np.ndarray, depth_bop: np.ndarray):
        """Run segmentation inference on a single RGB-D frame and always return a PIL image like visualize()."""
        rgb = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
        rgb_np = np.array(rgb)

        masks = self.model_segmentation.segmentor_model.generate_masks(rgb_np)
        if masks is None or len(masks) == 0:
            print("No masks detected!")
            return self.visualize_fallback(rgb, "No masks")

        print(f"{len(masks)} masks detected!")
        detections = Detections(masks)

        # Descriptor forward
        try:
            query_decriptors, query_appe_descriptors = self.model_segmentation.descriptor_model.forward(rgb_np, detections)
        except Exception as e:
            logging.warning(f"descriptor forward failed: {e}")
            return self.visualize_fallback(rgb, "Descriptor failed")

        # Semantic matching
        try:
            idx_selected_proposals, pred_idx_objects, semantic_score, best_template = \
                self.model_segmentation.compute_semantic_score(query_decriptors)
        except Exception as e:
            logging.warning(f"compute_semantic_score failed: {e}")
            return self.visualize_fallback(rgb, "Semantic failed")

        # Guard: nothing selected
        if idx_selected_proposals is None or len(idx_selected_proposals) == 0:
            return self.visualize_fallback(rgb, "No proposals")

        # Filter detections
        detections.filter(idx_selected_proposals)
        if getattr(detections, "masks", None) is None or len(detections) == 0:
            return self.visualize_fallback(rgb, "Empty after filter")

        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # Appearance score
        try:
            appe_scores, ref_aux_descriptor = self.model_segmentation.compute_appearance_score(
                best_template, pred_idx_objects, query_appe_descriptors
            )
        except Exception as e:
            logging.warning(f"compute_appearance_score failed: {e}")
            return self.visualize_fallback(rgb, "Appearance failed")

        # Geometry stage 
        try:
            batch = self.batch_input_data(depth_bop)

            template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
            template_poses[:, :3, 3] *= 0.4
            poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
            self.model_segmentation.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]

            mesh = trimesh.load_mesh(self.cad_path)
            model_points = mesh.sample(2048).astype(np.float32) / 1000.0
            self.model_segmentation.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(self.device)

            # Prevent N==1 squeezing issues by enforcing (N,H,W)
            if hasattr(detections, "masks") and detections.masks is not None:
                m = detections.masks
                if hasattr(m, "dim") and m.dim() == 2:
                    detections.masks = m.unsqueeze(0)
                elif isinstance(m, np.ndarray) and m.ndim == 2:
                    detections.masks = m[None, ...]

            image_uv = self.model_segmentation.project_template_to_image(
                best_template, pred_idx_objects, batch, detections.masks
            )

            geometric_score, visible_ratio = self.model_segmentation.compute_geometric_score(
                image_uv, detections, query_appe_descriptors, ref_aux_descriptor,
                visible_thred=self.model_segmentation.visible_thred
            )

        except Exception as e:
            logging.warning(f"Geometry failed/skipped: {e}")
            return self.visualize_fallback(rgb, "Geometry failed")

        # Final score
        final_score = (semantic_score + appe_scores + geometric_score * visible_ratio) / (1 + 1 + visible_ratio)
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))

        # Convert to list-of-dicts for visualize()
        detections.to_numpy()

        # If you still want to save, do it here (but consider not saving every frame)
        save_path = f"{self.output_dir}/sam6d_results/detection_ism"
        detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
        detections_json = convert_npz_to_json(idx=0, list_npz_paths=[save_path + ".npz"])
        save_json_bop23(save_path + ".json", detections_json)

        # Normal visualize output 
        vis_img = self.visualize(rgb, detections_json, save_path=None)
        return vis_img

    def run_pose_estimation_inference(self):
        """ Run pose estimation inference on a batch input data """
        def get_test_data(rgb_path, depth_path, cam_path, cad_path, seg_path, det_score_thresh, cfg):
            dets = []
            with open(seg_path) as f:
                dets_ = json.load(f) # keys: scene_id, image_id, category_id, bbox, score, segmentation
            for det in dets_:
                if det['score'] > det_score_thresh:
                    dets.append(det)
            del dets_

            cam_info = json.load(open(cam_path))
            K = np.array(cam_info['cam_K']).reshape(3, 3)

            whole_image = load_im(rgb_path).astype(np.uint8)
            if len(whole_image.shape)==2:
                whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
            whole_depth = load_im(depth_path).astype(np.float32) * cam_info['depth_scale'] / 1000.0
            whole_pts = get_point_cloud_from_depth(whole_depth, K)

            mesh = trimesh.load_mesh(cad_path)
            model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
            radius = np.max(np.linalg.norm(model_points, axis=1))


            all_rgb = []
            all_cloud = []
            all_rgb_choose = []
            all_score = []
            all_dets = []
            for inst in dets:
                seg = inst['segmentation']
                score = inst['score']

                # mask
                h,w = seg['size']
                try:
                    rle = cocomask.frPyObjects(seg, h, w)
                except:
                    rle = seg
                mask = cocomask.decode(rle)
                mask = np.logical_and(mask > 0, whole_depth > 0)
                if np.sum(mask) > 32:
                    bbox = get_bbox(mask)
                    y1, y2, x1, x2 = bbox
                else:
                    continue
                mask = mask[y1:y2, x1:x2]
                choose = mask.astype(np.float32).flatten().nonzero()[0]

                # pts
                cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
                center = np.mean(cloud, axis=0)
                tmp_cloud = cloud - center[None, :]
                flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
                if np.sum(flag) < 4:
                    continue
                choose = choose[flag]
                cloud = cloud[flag]

                if len(choose) <= cfg.n_sample_observed_point:
                    choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
                else:
                    choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
                choose = choose[choose_idx]
                cloud = cloud[choose_idx]

                # rgb
                rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
                if cfg.rgb_mask_flag:
                    rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
                rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
                rgb = rgb_transform(np.array(rgb))
                rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

                all_rgb.append(torch.FloatTensor(rgb))
                all_cloud.append(torch.FloatTensor(cloud))
                all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
                all_score.append(score)
                all_dets.append(inst)

            ret_dict = {}
            ret_dict['pts'] = torch.stack(all_cloud).cuda()
            ret_dict['rgb'] = torch.stack(all_rgb).cuda()
            ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
            ret_dict['score'] = torch.FloatTensor(all_score).cuda()

            ninstance = ret_dict['pts'].size(0)
            ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
            ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
            return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets

        print("=> loading input data ...")
        input_data, img, whole_pts, model_points, detections = get_test_data(
            "/home/nikolaraicevic/Workspace/External/SAM-6D/SAM-6D/Data/myObject/tomatoSoup/rgb.png", 
            "/home/nikolaraicevic/Workspace/External/SAM-6D/SAM-6D/Data/myObject/tomatoSoup/depth.png", 
            "/home/nikolaraicevic/Workspace/External/SAM-6D/SAM-6D/Data/myObject/tomatoSoup/camera.json", 
            self.cad_path, 
            os.path.join(self.output_dir, "sam6d_results", "detection_ism.json"),
            self.cfg.det_score_thresh, 
            self.cfg.test_dataset
        )
        ninstance = input_data['pts'].size(0)

        print("=> running model ...")
        with torch.no_grad():
            input_data['dense_po'] = self.all_tem_pts.repeat(ninstance,1,1)
            input_data['dense_fo'] = self.all_tem_feat.repeat(ninstance,1,1)
            out = self.pose_estimation_model(input_data)

        if 'pred_pose_score' in out.keys():
            pose_scores = out['pred_pose_score'] * out['score']
        else:
            pose_scores = out['score']
        pose_scores = pose_scores.detach().cpu().numpy()
        pred_rot = out['pred_R'].detach().cpu().numpy()
        pred_trans = out['pred_t'].detach().cpu().numpy() * 1000

        print("=> saving results ...")
        os.makedirs(f"{self.output_dir}/sam6d_results", exist_ok=True)
        for idx, det in enumerate(detections):
            detections[idx]['score'] = float(pose_scores[idx])
            detections[idx]['R'] = list(pred_rot[idx].tolist())
            detections[idx]['t'] = list(pred_trans[idx].tolist())

        with open(os.path.join(f"{self.output_dir}/sam6d_results", 'detection_pem.json'), "w") as f:
            json.dump(detections, f)

        print("=> visualizating ...")
        save_path = os.path.join(f"{self.output_dir}/sam6d_results", 'vis_pem.png')
        valid_masks = pose_scores == pose_scores.max()
        K = input_data['K'].detach().cpu().numpy()[valid_masks]
        vis_img = self.visualize_pose_estimation(img, pred_rot[valid_masks], pred_trans[valid_masks], model_points*1000, K, save_path)
        vis_img.save(save_path)


# Main
def main():
    parser = argparse.ArgumentParser(description="Live SAM-6D inference from RealSense stream.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--cad_path", type=str, required=True, help="Path to CAD model in mm units (e.g., obj_000004.ply).")
    # Segmentor parameters
    parser.add_argument("--segmentor_model", default="sam", choices=["sam", "fastsam"], help="Segmentor model to use.")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    # Pose Estimation parameters
    parser.add_argument("--det_score_thresh", default=0.2, type=float, help="The score threshold of detection")
    parser.add_argument("--gpus", type=str, default="0", help="path to pretrain model")
    parser.add_argument("--pose_estimation_model", type=str, default="pose_estimation_model", help="path to model file")
    parser.add_argument("--iter", type=int, default=600000, help="epoch num. for testing")
    parser.add_argument("--exp_id", type=int, default=0, help="")
    args = parser.parse_args()

    # Initialize RealSense Camera
    realsense = RealSenseCamera(
        out_dir=args.output_dir, 
        depth_scale=1.0, 
        intrinsics_for="color", 
        align_to_color=True)
    camera_intrinsics = realsense.get_camera_intrinsics(save_json=True, print_info=True)
    print("Camera intrinsics BOP:\n", camera_intrinsics["bop"])
    
    # Initialize Object Tracker
    tracker = ObjectTracker(
        output_dir=args.output_dir,
        cad_path=args.cad_path,
        cam_K=camera_intrinsics["bop"]["cam_K"],
        depth_scale = camera_intrinsics["bop"]["depth_scale"],
        segmentor_model=args.segmentor_model,
        stability_score_thresh=args.stability_score_thresh,
        det_score_thresh=args.det_score_thresh,
        gpus=args.gpus,
        pose_estimation_model=args.pose_estimation_model,
        iter=args.iter,
        exp_id=args.exp_id,
    )

    print("Test pose estimation inference ...")
    tracker.run_pose_estimation_inference()

    # window_name = "SAM-6D Live (q to quit)"
    # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # try:
    #     for color_bgr, depth_bop in realsense.frames():
    #         vis_img = tracker.run_segmentation_inference(color_bgr, depth_bop)
    #         cv2.imshow(window_name, cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR))

    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord("q"):
    #             break
    #         elif key == ord("s"):
    #             out_dir = os.path.join(args.output_dir, "sam6d_results")
    #             os.makedirs(out_dir, exist_ok=True)
    #             save_path = os.path.join(out_dir, f"vis_ism.png")
    #             vis_img.save(save_path)
    #             print(f"Saved visualization to {save_path}")
    # finally:
    #     cv2.destroyAllWindows()
    #     del realsense


if __name__ == "__main__":
    main()
