# AGC-Drive: A Large-Scale Dataset for Real-World Aerial-Ground Collaboration in Driving Scenarios

**AGC-Drive** is a large-scale, real-world dataset developed to advance autonomous driving research with aerial-ground collaboration. It enables multi-agent information sharing to overcome challenges such as occlusion and limited perception range, improving perception accuracy in complex driving environments.

While existing datasets often focus on vehicle-to-vehicle (V2V) or vehicle-to-infrastructure (V2I) collaboration, **AGC-Drive innovatively incorporates aerial views from unmanned aerial vehicles (UAVs)**. This integration provides dynamic, top-down perspectives that effectively reduce occlusion issues and allow monitoring of large-scale interactive scenarios.


---

## 📦 Dataset Overview

The dataset was collected using a collaborative sensing platform consisting of:

- **Two vehicles**, each equipped with **5 cameras and 1 LiDAR sensor**  
- **One UAV**, equipped with a **forward-facing camera and a LiDAR sensor**

It includes:

- **~120K LiDAR frames**  
- **~440K images**  
- **14 diverse real-world driving scenarios** (e.g., urban roundabouts, highway tunnels, on/off ramps)  
- **400 scenes**, each with approximately **100 frames**  
- Fully annotated **3D bounding boxes for 13 object categories**  
- **19.5% of frames** featuring dynamic interaction events: cut-ins, cut-outs, frequent lane changes

An open-source toolkit is also provided, featuring:

- 🗺️ Spatiotemporal alignment verification tools  
- 📊 Multi-agent collaborative visualization systems  
- 📝 Collaborative 3D annotation utilities  

---

## 📥 Download Dataset

We provide two download options:

- 📦 **Baidu Netdisk (Recommended for fast download in China)**  
  🔗 [**Download via Baidu Netdisk**](https://pan.baidu.com/s/13r7msTs196CpG9huTyoRYQ?pwd=yen6)  
  (Password: `yen6`)

- 🌐 **Public Web Preview & Download (Supports online browsing, slower download)**  
  🔗 [**Download via Public Link**](http://49.232.218.165:5911/)

---

## 📝 Data Collection Method

Data was gathered across various urban and highway driving scenarios with hardware-level time synchronization and precise sensor calibration. It includes multi-agent LiDAR, multi-view RGB images, GPS/IMU data, and annotated 3D bounding boxes for collaborative perception applications.

---

## 📊 Benchmark Methods

We evaluate AGC-Drive with the following baseline models:

| Method             | Type                     | 3D Detection | Cooperative Fusion | Description |
|:------------------|:-------------------------|:-------------|:-------------------|:-----------------------------------------------------------|
| **Upper-bound**     | Early Fusion              | ✅           | ✅                  | Shares raw point cloud data before feature extraction.      |
| **Lower-bound**     | Late Fusion               | ✅           | ✅                  | Independently detects and shares detection results.         |
| **V2VNet** [1]      | Intermediate Fusion       | ✅           | ✅                  | Multi-agent detection via intermediate feature fusion.      |
| **CoBEVT** [28]     | Intermediate Fusion (BEV) | ✅           | ✅                  | Sparse Transformer BEV fusion with FAX module.              |
| **Where2comm** [13] | Communication-efficient   | ✅           | ✅                  | Shares sparse, critical features guided by confidence maps. |
| **V2X-ViT** [4]     | Transformer-based Fusion  | ✅           | ✅                  | BEV feature fusion via attention mechanisms.                |

---

## 🐍 Python Environment Setup

Recommended: **Python 3.7+**, **CUDA 11.7+**

### Install via Conda:
```bash
conda env create -f environment.yml
conda activate agcdrive
```

## 📚 Supported Projects

The following key projects and papers are referenced and used as baselines in our benchmarks:

- **V2VNet**  
  Runsheng Xu, Hao Xiang, Xin Xia, Xu Han, Jinlong Li, and Jiaqi Ma. Opv2v: An open benchmark dataset
  and fusion pipeline for perception with vehicle-to-vehicle communication. In 2022 International Conference on
  Robotics and Automation (ICRA), page 2583–2589. IEEE Press, 2022.  
  [Paper](https://arxiv.org/abs/2008.07519)

- **CoBEVT**  
  Hao Xiang Wei Shao Bolei Zhou Jiaqi Ma Runsheng Xu, Zhengzhong Tu. Cobevt: Cooperative bird’s eye
view semantic segmentation with sparse transformers. In Conference on Robot Learning (CoRL), 2022.  
  [Paper](https://openreview.net/forum?id=PAFEQQtDf8s)

- **Where2comm**  
  Yue Hu, Shaoheng Fang, Zixing Lei, Yiqi Zhong, and Siheng Chen. Where2comm: Communication-
efficient collaborative perception via spatial confidence maps. Advances in neural information processing
systems, 35:4874–4886, 2022.  
  [Paper](https://openreview.net/forum?id=dLL4KXzKUpS)

- **V2X-ViT**  
  Runsheng Xu et al. V2x-vit: Vehicle-to-everything cooperative perception with vision transformer. In ECCV Proceedings, 2022.  
  [Paper](https://arxiv.org/abs/2203.10638)

