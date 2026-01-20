# <p align="center"> <font color=#008000>SAM-6D</font>: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation </p>

####  <p align="center"> [Jiehong Lin](https://jiehonglin.github.io/), [Lihua Liu](https://github.com/foollh), [Dekun Lu](https://github.com/WuTanKun), [Kui Jia](http://kuijia.site/)</p>
#### <p align="center">CVPR 2024 </p>
#### <p align="center">[[Paper]](https://arxiv.org/abs/2311.15707) </p>

<p align="center">
  <img width="100%" src="https://github.com/JiehongLin/SAM-6D/blob/main/pics/vis.gif"/>
</p>


## News
- [2024/03/07] We publish an updated version of our paper on [ArXiv](https://arxiv.org/abs/2311.15707).
- [2024/02/29] Our paper is accepted by CVPR2024!


## Update Log
- [2024/03/05] We update the demo to support [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), you can do this by specifying `SEGMENTOR_MODEL=fastsam` in demo.sh.
- [2024/03/03] We upload a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for running custom data.
- [2024/03/01] We update the released [model](https://drive.google.com/file/d/1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7/view?usp=sharing) of PEM. For the new model, a larger batchsize of 32 is set, while that of the old is 12. 

## Overview
In this work, we employ Segment Anything Model as an advanced starting point for **zero-shot 6D object pose estimation** from RGB-D images, and propose a novel framework, named **SAM-6D**, which utilizes the following two dedicated sub-networks to realize the focused task:
- [x] [Instance Segmentation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Instance_Segmentation_Model)
- [x] [Pose Estimation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Pose_Estimation_Model)


<p align="center">
  <img width="50%" src="https://github.com/JiehongLin/SAM-6D/blob/main/pics/overview_sam_6d.png"/>
</p>


## Getting Started

### 1. Preparation
Please clone the repository locally:
```
git clone https://github.com/JiehongLin/SAM-6D.git
```
Install the environment and download the model checkpoints:
```
cd SAM-6D
sh prepare.sh
```
We also provide a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for convenience.

### 2. Evaluation on the custom data
```
# set the paths
export CAD_PATH=$(pwd)/Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=$(pwd)/Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=$(pwd)/Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=$(pwd)/Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=$(pwd)/Data/Example/outputs         # path to a pre-defined file for saving results

# run inference
sh demo.sh
```

### 3. Run new object
```
# ============================================
# Configuration - EDIT THESE PATHS
# ============================================
OBJECT_DIR="myObject/banana"    # Directory containing your data
CAD_FILE="obj_000010.ply"           # Your CAD model filename
RGB_FILE="rgb.png"                  # Your RGB image filename
DEPTH_FILE="depth.png"              # Your depth image filename
CAMERA_FILE="camera.json"           # Your camera intrinsics filename

# ============================================
# Set Paths
# ============================================
export CAD_PATH=$(pwd)/Data/${OBJECT_DIR}/${CAD_FILE}
export RGB_PATH=$(pwd)/Data/${OBJECT_DIR}/${RGB_FILE}
export DEPTH_PATH=$(pwd)/Data/${OBJECT_DIR}/${DEPTH_FILE}
export CAMERA_PATH=$(pwd)/Data/${OBJECT_DIR}/${CAMERA_FILE}
export OUTPUT_DIR=$(pwd)/Data/${OBJECT_DIR}/outputs

# ============================================
# Run Pipeline
# ============================================
echo "========================================="
echo "Running SAM-6D Pipeline"
echo "========================================="
echo "CAD Model: $CAD_PATH"
echo "RGB Image: $RGB_PATH"
echo "Depth Image: $DEPTH_PATH"
echo "Camera: $CAMERA_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Run the pipeline
sh demo.sh

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "Results saved to: $OUTPUT_DIR/sam6d_results/"
echo "========================================="
```



## Citation
If you find our work useful in your research, please consider citing:

    @article{lin2023sam,
    title={SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation},
    author={Lin, Jiehong and Liu, Lihua and Lu, Dekun and Jia, Kui},
    journal={arXiv preprint arXiv:2311.15707},
    year={2023}
    }


## Contact

If you have any questions, please feel free to contact the authors. 

Jiehong Lin: [mortimer.jh.lin@gmail.com](mailto:mortimer.jh.lin@gmail.com)

Lihua Liu: [lihualiu.scut@gmail.com](mailto:lihualiu.scut@gmail.com)

Dekun Lu: [derkunlu@gmail.com](mailto:derkunlu@gmail.com)

Kui Jia:  [kuijia@gmail.com](kuijia@gmail.com)

