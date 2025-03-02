Vehicle Height Estimation using YOLO12 and DepthAnythingV2
==========================================================

### Overview

This demo project implements an automated vehicle height estimation pipeline. The system processes vehicle footage from YouTube by leveraging object detection, tracking, and segmentation with YOLO, depth estimation using DepthAnythingV2, and additional image processing to calculate the vehicle’s height in real-world units. The pipeline is integrated into a two-page Streamlit application:
- Tracking & Segmentation: Streams and displays detection frames in real time while saving tracked vehicles.
- Results: Displays combined visualization images and log data (CSV) for further analysis. 

---

### Sample WebApp Screenshots
- **Inference Page**
![VHEScreenShot1](https://github.com/user-attachments/assets/496ee678-6a5d-4e9d-8e9e-11c74f578c62)
- **Results Page**
![VHEScreenShot2](https://github.com/user-attachments/assets/3732037f-f879-4fd1-aff7-16df08ee4a8f)
![VHEScreenShot3](https://github.com/user-attachments/assets/6e01dd59-7d9e-408c-b41c-895eae5d19ac)


### Project Highlights

#### **Core Technologies**
- **YOLOv12 & YOLO11x-seg** - Performs object detection, tracking, and segmentation of vehicles from YouTube video streams.
- **DepthAnythingV2** - Provides high-quality depth estimation from 2D images.
- **OpenCV** - Handles image processing, segmentation overlays, cropping, and visualization.
- **Streamlit** - Creates an interactive two-page web UI that allows users to input a YouTube link and view processing results in real time.

#### Hardware for Inference
- Nvidia GeForce RTX 4070 Mobile GPU
- Ryzen 9 AI HX 370

#### Operating System:
- Windows 11

#### **Main Components**
- **track_and_segment.py** - Detects and tracks vehicles from YouTube videos, saves unannotated frames, segmentation masks, and cropped images.
- **estimate_depth.py** - Performs depth estimation on saved frames using DepthAnythingV2.
- **calculate.py** - Calculates vehicle height in real-world units using segmentation masks and depth information.
- **Inference.py** - Main Streamlit app that runs the pipeline and streams frames in real time.
- **Results.py** - A separate Streamlit page displaying combined visualizations and logs.

#### **Automation Scripts**
- **run_init.ps1** - Powershell script that automates the project initialization, including the creating of virtual env, addition of depthanythingv2 as submodule, and installation of dependencies.
---



### **Launching the App**
#### A. Locally
1. Install dependencies:
  ```powershell
  ./run_init.ps1
  ```
2. Open Terminal and issue the command:
  ```powershell
  streamlit run Inference.py
  ```

#### B. Using Docker
1. Build the Image - to create a Docker image, run:
  ```powershell
  docker build -t vehicle-height-estimator .
  ```
This will install all required dependencies and set up the Streamlit app inside the container.

2. Running the Container
To run the container and expose the Streamlit app on port 8501 (this will also utilize GPU):
  ```powershell
  docker run --gpus all -p 8501:8501 vehicle-height-estimator 
  ```

3. Open browser and go to:
  ```powershell
  http://localhost:8501
  ```

### Credits
- **Ultralytics** for the YOLO tracking/detection/segmentation models.
- **DepthAnythingV2** for depth estimation.
---
