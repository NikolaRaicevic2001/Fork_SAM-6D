#!/usr/bin/env python3
import os
import cv2
import sys
import glob
import torch
import trimesh
import imageio
import logging
import argparse
import distinctipy
import numpy as np

from tqdm import tqdm
from PIL import Image
from skimage.feature import canny
from hydra.utils import instantiate
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

# Define repository paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ISM_ROOT = os.path.join(THIS_DIR, "Instance_Segmentation_Model")
CKPT_ROOT = os.path.join(ISM_ROOT, "checkpoints")

if ISM_ROOT not in sys.path:
    sys.path.insert(0, ISM_ROOT)

from camera import RealSenseCamera
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad
from Instance_Segmentation_Model.utils.inout import load_json, save_json_bop23
from Instance_Segmentation_Model.model.utils import Detections, convert_npz_to_json
from Instance_Segmentation_Model.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2

logging.basicConfig(level=logging.INFO)

# ObjectTracker
class ObjectTracker:
    def __init__( self, segmentor_model: str, output_dir: str, cad_path: str, cam_K: np.ndarray, depth_scale: float, stability_score_thresh: float = 0.97):
        """ Initialize the Object Tracker with segmentation and descriptor models """
        # Initialize parameters
        self.segmentor_model = segmentor_model
        self.output_dir = output_dir
        self.cad_path = cad_path
        self.cam_K = cam_K
        self.depth_scale = depth_scale
        self.stability_score_thresh = stability_score_thresh

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
        self.model = instantiate(cfg.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.descriptor_model.model = self.model.descriptor_model.model.to(self.device)
        self.model.descriptor_model.model.device = self.device

        # if there is predictor in the model, move it to device
        if hasattr(self.model.segmentor_model, "predictor"):
            self.model.segmentor_model.predictor.model = (self.model.segmentor_model.predictor.model.to(self.device))
        else:
            self.model.segmentor_model.model.setup_model(device=self.device, verbose=True)
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

        self.model.ref_data = {}
        self.model.ref_data["descriptors"] = self.model.descriptor_model.compute_features(templates, token_name="x_norm_clstoken").unsqueeze(0).data
        self.model.ref_data["appe_descriptors"] = self.model.descriptor_model.compute_masked_patch_feature(templates, masks_cropped[:, 0, :, :]).unsqueeze(0).data
        
    def batch_input_data(self, depth_path, device):
        batch = {}
        depth = np.array(imageio.imread(depth_path)).astype(np.int32)
        cam_K = np.array(self.cam_K).reshape((3, 3))
        depth_scale = np.array(self.depth_scale)

        batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
        batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
        batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
        return batch
    
    def visualize(self, rgb, detections, save_path="tmp.png"):
        img = rgb.copy()
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        colors = distinctipy.get_colors(len(detections))
        alpha = 0.33

        best_score = 0.
        for mask_idx, det in enumerate(detections):
            if best_score < det['score']:
                best_score = det['score']
                best_det = detections[mask_idx]

        mask = rle_to_mask(best_det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = best_det["category_id"]
        temp_id = obj_id - 1

        r = int(255*colors[temp_id][0])
        g = int(255*colors[temp_id][1])
        b = int(255*colors[temp_id][2])
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
        img[edge, :] = 255
        
        img = Image.fromarray(np.uint8(img))
        img.save(save_path)
        prediction = Image.open(save_path)
        
        # concat side by side in PIL
        img = np.array(img)
        concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
        concat.paste(rgb, (0, 0))
        concat.paste(prediction, (img.shape[1], 0))
        return concat

    def run_segmentation_inference(self, rgb_path: str, depth_path: str):
        """ Run segmentation inference on a single RGB-D frame """
        rgb = Image.open(rgb_path).convert("RGB")
        detections = self.model.segmentor_model.generate_masks(np.array(rgb))
        detections = Detections(detections)
        query_decriptors, query_appe_descriptors = self.model.descriptor_model.forward(np.array(rgb), detections)

        # matching descriptors
        (idx_selected_proposals, pred_idx_objects, semantic_score, best_template,) = self.model.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor= self.model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

        # compute the geometric score
        batch = self.batch_input_data(depth_path, self.device)
        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
        self.model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

        mesh = trimesh.load_mesh(self.cad_path)
        model_points = mesh.sample(2048).astype(np.float32) / 1000.0
        self.model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(self.device)
        
        image_uv = self.model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

        geometric_score, visible_ratio = self.model.compute_geometric_score(image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=self.model.visible_thred)

        # final score
        final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))   
            
        detections.to_numpy()
        save_path = f"{self.output_dir}/sam6d_results/detection_ism"
        detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
        detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
        save_json_bop23(save_path+".json", detections)
        vis_img = self.visualize(rgb, detections, f"{self.output_dir}/sam6d_results/vis_ism.png")
        vis_img.save(f"{self.output_dir}/sam6d_results/vis_ism.png")
        print(f"Segmentation results saved to {self.output_dir}/sam6d_results/")


# Main
def main():
    parser = argparse.ArgumentParser(description="Live SAM-6D inference from RealSense stream.")
    parser.add_argument("--segmentor_model", default="sam", choices=["sam", "fastsam"], help="Segmentor model to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--cad_path", type=str, required=True, help="Path to CAD model in mm units (e.g., obj_000004.ply).")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
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
        segmentor_model=args.segmentor_model,
        output_dir=args.output_dir,
        cad_path=args.cad_path,
        cam_K=camera_intrinsics["bop"]["cam_K"],
        depth_scale = camera_intrinsics["bop"]["depth_scale"],
        stability_score_thresh=args.stability_score_thresh,
    )

    print("Try running inference on a single frame...")
    tracker.run_segmentation_inference(
        rgb_path="/home/nikolaraicevic/Workspace/External/SAM-6D/SAM-6D/Data/myObject/tomatoSoup/rgb.png",
        depth_path="/home/nikolaraicevic/Workspace/External/SAM-6D/SAM-6D/Data/myObject/tomatoSoup/depth.png",
    )

    # try:
        
    # finally:
    #     del realsense






    # window_name = "SAM-6D Live (q to quit)"
    # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # try:
    #     for color_bgr, depth_bop in cam.frames():
    #         rgb_pil, detections = tracker.run_segmentation_inference(color_bgr, depth_bop)
    #         vis_bgr = visualize_best(rgb_pil, detections)
    #         cv2.imshow(window_name, vis_bgr)

    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord("q"):
    #             break
    # finally:
    #     cv2.destroyAllWindows()
    #     del cam


if __name__ == "__main__":
    main()
