# 🚗 Traffic Analysis 

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Dependencies and Imports](#dependencies-and-imports)
3. [Path and Configuration Setup](#path-and-configuration-setup)
4. [Model and Tracker Initialization](#model-and-tracker-initialization)
5. [Helper Functions](#helper-functions)
6. [Main Processing Function](#main-processing-function)
7. [Output Generation](#output-generation)
8. [Script Execution](#script-execution)

## 1. Introduction <a name="introduction"></a>

This script is designed for advanced traffic analysis using computer vision techniques. It processes video footage of traffic, detecting and tracking vehicles, analyzing their movements, and generating various visualizations and statistics. The script uses the YOLOv10 object detection model and the OC-SORT tracking algorithm to achieve this.

## 2. Dependencies and Imports <a name="dependencies-and-imports"></a>

The script relies on several Python libraries:

- 🐍 `os`: For file and directory operations
- 🖼️ `cv2`: OpenCV for image and video processing
- 🧮 `numpy`: For numerical operations
- 🔥 `torch`: PyTorch for deep learning operations
- 🗺️ `shapely`: For geometric operations
- 📊 `matplotlib`: For color mapping
- 🔁 `itertools`: For creating cycles of colors
- 🕵️ `ultralytics`: For the YOLOv10 model
- 🔍 `boxmot`: For the OC-SORT tracker

## 3. Path and Configuration Setup <a name="path-and-configuration-setup"></a>

The script sets up various paths and configurations:

- 📁 Input video path
- 📂 Output folder for generated files
- 🎥 Output paths for different types of analysis videos
- 🎨 Color configurations for object classes and ROI lines
- 📍 Definition of critical positions for object counting

## 4. Model and Tracker Initialization <a name="model-and-tracker-initialization"></a>

- 🧠 The YOLOv10 model is loaded with a custom weights file (`best_v10l.pt`)
- 🔎 The OC-SORT tracker is initialized
- 💻 The script checks for GPU availability and sets the device accordingly

## 5. Helper Functions <a name="helper-functions"></a>

The script includes several helper functions:

- 🔍 `detect_objects`: Uses YOLOv10 to detect objects in a frame
- 🔭 `track_objects`: Uses OC-SORT to track detected objects
- ✅ `crosses_threshold`: Checks if an object crosses a defined threshold
- 📍 `is_inside_critical_location`: Checks if an object is inside a defined area
- 🖌️ `draw_blinking_count_roi`: Draws ROI lines with a blinking effect
- 📝 `draw_info_on_canvas`: Draws object counts and ROI lines on a canvas
- 🌈 `get_unique_color`: Generates unique colors for object trails
- 🖼️ `plot_trail`: Plots object trails on a canvas
- 🔗 `combine_frames`: Combines multiple frames into a single image
- 🏎️ `estimate_speed`: Estimates object speed based on position changes
- 🔥 `create_heatmap`: Creates a heatmap visualization
- 📊 `visualize_speed_areas`: Creates a visualization of speed areas
- 🔢 `draw_counts_near_roi`: Draws object counts near ROI lines

## 6. Main Processing Function <a name="main-processing-function"></a>

The `process_video` function is the core of the script:

1. 📂 It opens the input video and initializes video writers for various output videos
2. 🔄 For each frame in the video:
   - 🔍 Detects objects using YOLOv10
   - 🔭 Tracks objects using OC-SORT
   - 📏 Calculates object positions and speeds
   - 🔢 Counts objects crossing defined ROI lines
   - 🖌️ Draws bounding boxes, trails, and information on the frame
   - 🔥 Creates heatmaps for speed and object count
   - 📊 Generates visualizations for speed areas
3. 💾 Writes processed frames to output videos

## 7. Output Generation <a name="output-generation"></a>

The script generates several output videos:

- 🎥 Main output video (2_results.mp4): Shows object detection, tracking, and counting
- 🌈 Trail video (2_trails.mp4): Displays object movements over time
- 🔥 Speed heatmap video (2_speed_heatmap.mp4): Visualizes speed areas
- 🚀 Speed dots video (2_speed_dots.mp4): Shows objects with color-coded speed indicators
- 🖼️ Trail combo video (2_trail_combo.mp4): Combines the main output and trail video side by side
- 📊 Master analysis video (2_master_analysis.mp4): Combines all visualizations into a single video

## 8. Script Execution <a name="script-execution"></a>

The script is executed by calling the `process_video` function with the defined input and output paths.

## 🔑 Key Features

1. 🚗 Multi-object detection and tracking
2. 🏎️ Speed estimation for tracked objects
3. 🔢 Object counting across defined ROI lines
4. 🌈 Object trail visualization
5. 🔥 Speed and count heatmap generation
6. 📊 Speed area visualization
7. 🖼️ Combined master analysis video

## 📥 Inputs

- 🎥 Input video file
- 🧠 YOLOv10 model weights file
- 📍 Predefined critical positions and ROI lines

## 📤 Outputs

- 🎥 Main output video (2_results.mp4) with object detection, tracking, and counting
- 🌈 Trail video (2_trails.mp4)
- 🔥 Speed heatmap video (2_speed_heatmap.mp4)
- 🚀 Speed dots video (2_speed_dots.mp4)
- 🖼️ Trail combo video (2_trail_combo.mp4)
- 📊 Master analysis video (2_master_analysis.mp4) combining all visualizations

## 🔧 How It Works

1. 📂 The script reads the input video frame by frame
2. 🔍 For each frame, it detects objects using YOLOv10
3. 🔭 Detected objects are tracked across frames using OC-SORT
4. 🏎️ Object speeds are estimated based on position changes
5. 🔢 Objects crossing predefined ROI lines are counted
6. 🖌️ Various visualizations are drawn on the frame:
   - Bounding boxes around detected objects
   - Object trails
   - Speed and count information
   - ROI lines
7. 🔥 Heatmaps are generated for speed and object count
8. 📊 Speed areas are visualized
9. 🖼️ All visualizations are combined into a master analysis frame
10. 💾 Processed frames are written to output videos

This script provides a comprehensive solution for traffic analysis, offering valuable insights into vehicle movements, speeds, and counts in different areas of the monitored region. The various visualizations make it easy to understand complex traffic patterns at a glance.

# Video Processing Flowchart

```mermaid
graph TD
    A[Start] --> B[Initialize YOLO model and OC-SORT tracker]
    B --> C[Open input video]
    C --> D[Initialize video writers for output]
    D --> E[Enter main processing loop]
    E --> F[Read frame from video]
    F --> G{End of video?}
    G -->|No| H[Detect objects in frame]
    H --> I[Track objects]
    I --> J[Draw blinking count ROI]
    J --> K[Process each tracked object]
    K --> L[Estimate object speed]
    L --> M[Check if object crosses critical positions]
    M --> N[Update object counts]
    N --> O[Create speed dots visualization]
    O --> P[Create speed area visualization]
    P --> Q[Draw trails if enabled]
    Q --> R[Combine frames for master analysis]
    R --> S[Write frames to output videos]
    S --> E
    G -->|Yes| T[Release video captures and writers]
    T --> U[End]
