## Automatic Target-Free LiDAR-Camera Extrinsic Calibration via Multi-Modal Geometric Edge Feature Extraction and Visibility Analysis

This method is a target-free calibration technique for multimodal sensors, whose core advantage lies in achieving fully automatic extrinsic parameter estimation from a single frame of LiDAR and camera data. It addresses key limitations of traditional approaches, such as their reliance on manual calibration targets, the susceptibility of line feature extraction to texture interference, and the neglect of occluded features during optimization. By introducing dual innovations in multimodal edge feature extraction and feature visibility analysis, the method significantly enhances calibration applicability and accuracy.

### Key Innovations:

- A Novel Point Cloud Feature Extraction Algorithm: A dedicated feature extractor is proposed, which employs a local feature decomposition strategy to effectively identify and acquire salient boundary points from the point cloud.

- A Novel Image Feature Extraction Algorithm: This algorithm achieves the extraction of 3D edge features from 2D image by integrating traditional 2D image feature detection with a learning-based depth estimation strategy.

- A Novel Point Cloud Visibility Analysis Model: This model assesses the visibility of each point by analyzing the Gaussian distribution and the normal vector consistency, effectively identifying and removing the influence of occluded points during optimization.

## Framework
After independent extraction of geometric edge features from both point clouds and image, a visibility analysis module filters out occluded points; an elliptical matching strategy then establishes cross-modal correspondences, followed by optimization of extrinsic parameters **[R, t]** through reprojection error minimization.
<div style="display: flex; gap: 10px; flex-wrap: wrap;">
  <img src="doc/OV.png" alt="pipeline" style="max-width: 80%; height: auto;"/>
</div>

## Experimental Setup & Dataset collection
The LiDAR-camera experimental setup, featuring a Livox-AVIA LiDAR and a HIKROBOT MV-CA013-21UM camera.The relative pose between the sensors remained fixed throughout data collection. Please note the coordinate transformation relationships between sensors.
<div style="display: flex; gap: 10px; flex-wrap: wrap;">
  <img src="doc/setup.png" alt="pipeline" width="50%" align="center"/>
</div>

## 1. Prerequisites
- **Ubuntu** (tested on Ubuntu 18.04/20.04)
- **PCL**(>=1.8) [Installation](http://www.pointclouds.org/downloads/linux.html).
- **OpenCV**(==3.x) [Installation](https://github.com/opencv/opencv/tags).
- **Eigen**(>=3.3) [Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page).
- **Ceres**(==1.4) [Installation](http://ceres-solver.org/installation.html).

## 2. Compilation & Run
Clone the code and compilation: adjust the parameters in `config.yaml` according to your data and launch environment before compilation or execution.
```
    git clone https://github.com/xxx/calibration_LiDAR_camera.git
    cd calibration_LiDAR_camera
    mkdir build && cd build
    cmake ..
    make
    ./calib
```
## 3. Calibration Results
The raw image, overlaping image, and colorized points.
<div style="display: flex; gap: 10px; flex-wrap: wrap;">
  <img src="doc/raw.png" alt="pipeline" width="30%"/>
  <img src="doc/overlaping.png" alt="pipeline" width="30%""/>
  <img src="doc/colorized_points.gif" alt="pipeline" width="30%"/>
</div>


## 4.Acknowledgments
We sincerely appreciate the following open-source projects: [Pandey](https://github.com/xmba15/automatic_lidar_camera_calibration.git), [livox_camera_calib](https://github.com/hku-mars/livox_camera_calib.git), [Koide](https://github.com/koide3/direct_visual_lidar_calibration), and [MFCalib](https://github.com/Es1erda/MFCalib.git).

## License
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.
