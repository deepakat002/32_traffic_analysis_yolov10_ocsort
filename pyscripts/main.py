import os
import cv2
import sys
import numpy as np
import torch
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
from itertools import cycle

# Ensure the boxmot folder is in the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'boxmot'))
from ultralytics import YOLOv10, YOLO
from boxmot.trackers.ocsort.ocsort import OCSort
from helper_utils import *

# Define paths
video_path = '/trafficyolo/videos/2.mp4'  # Update this path
output_folder = '/trafficyolo/output'
create_trail = True

# Get the output file names based on the input file name
input_file_name = os.path.basename(video_path)
input_file_name_no_ext = os.path.splitext(input_file_name)[0]
output_video_path = os.path.join(output_folder, f"{input_file_name_no_ext}_results.mp4")
trail_video_path = os.path.join(output_folder, f"{input_file_name_no_ext}_trails.mp4")
speed_heatmap_path = os.path.join(output_folder, f"{input_file_name_no_ext}_speed_heatmap.mp4")
master_analysis_path = os.path.join(output_folder, f"{input_file_name_no_ext}_master_analysis.mp4")

# Create necessary folders
os.makedirs(output_folder, exist_ok=True)

# Check if GPU is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize YOLOv10 model and OC-SORT tracker
model = YOLOv10('/trafficyolo/model_weights/best_v10l.pt').to(device)
model.conf = 0.75
tracker = OCSort()
print(model.names)

# Dictionary to keep track of object counts
crossed_objects = {f'position{i+1}': 0 for i in range(4)}
counted_objects = {f'position{i+1}': set() for i in range(4)}


##### ------------------------------------- main workflow -----------------------------------------------

def process_video(video_path, output_video_path, trail_video_path, speed_heatmap_path, master_analysis_path, create_trail):
    """
    This function processes the input video, performs object detection and tracking,
    and generates various output videos with analytics
    - input: video_path (string), output_video_path (string), trail_video_path (string),
             speed_heatmap_path (string), master_analysis_path (string), create_trail (boolean)
    - output: None (generates output video files)
    """


    print("Starting video processing...")
    
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    print(f"Video opened: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")

    # Initialize video writers for different output videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    trail_out = cv2.VideoWriter(trail_video_path, fourcc, fps, (frame_width, frame_height))
    speed_dots_path = os.path.join(output_folder, f"{input_file_name_no_ext}_speed_dots.mp4")
    speed_dots_out = cv2.VideoWriter(speed_dots_path, fourcc, fps, (frame_width, frame_height))
    speed_area_out = cv2.VideoWriter(speed_heatmap_path, fourcc, fps, (frame_width, frame_height + 150))
    master_out = cv2.VideoWriter(master_analysis_path, fourcc, fps, (frame_width * 2, frame_height * 2))
    combo_video_path = os.path.join(output_folder, f"{input_file_name_no_ext}_trail_combo.mp4")
    combo_out = cv2.VideoWriter(combo_video_path, fourcc, fps, (frame_width * 2, frame_height))
    print("Video writers initialized")

    # Initialize variables for tracking and analysis
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    trails = {}
    trail_colors = {}
    grid_counts = np.zeros((frame_height // 40, frame_width // 40), dtype=int)
    crossed_objects = {pos: 0 for pos in critical_positions}
    counted_objects = {pos: set() for pos in critical_positions}
    blink_state = {pos: False for pos in critical_positions}
    blink_frames = {pos: 0 for pos in critical_positions}
    last_positions = {pos: {} for pos in critical_positions}
    object_speeds = {}

    frame_count = 0
    grid_overlay_saved = False
    print("Starting main processing loop...")

    # Main video processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Save grid overlay for the first frame
        if not grid_overlay_saved:
            grid_overlay_path = os.path.join(output_folder, f"{input_file_name_no_ext}_grid_overlay.jpg")
            draw_grid_overlay(frame, grid_size=40, output_path=grid_overlay_path)
            grid_overlay_saved = True
        # Detect objects in the current frame
        detections = detect_objects(frame)
        
        # Track detected objects
        tracked_objects = track_objects(frame, detections)

        # Draw blinking count ROI on the frame
        draw_blinking_count_roi(frame, blink_state)

        # speed_heatmap_data = [] 
        # Process each tracked object
        for obj in tracked_objects:
            # Extract object information
            x1, y1, x2, y2, obj_id, _, cls, _ = obj
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)

            # Draw bounding box and centroid
            color = class_colors.get(cls, (0, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (centroid_x, centroid_y), 3, color, -1)

            # Estimate object speed
            if obj_id in object_speeds:
                speed = estimate_speed(object_speeds[obj_id]['prev_pos'], (centroid_x, centroid_y), fps)
                object_speeds[obj_id]['speed'] = speed
                object_speeds[obj_id]['prev_pos'] = (centroid_x, centroid_y)
            else:
                object_speeds[obj_id] = {'prev_pos': (centroid_x, centroid_y), 'speed': 0}

            speed = object_speeds[obj_id]['speed']
            cv2.putText(frame, f"{model.names[int(cls)]}:{obj_id}/{speed:.1f}kmph", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # speed_heatmap_data.append(((centroid_x, centroid_y), speed))

            # Check if object crosses critical positions
            for pos, data in critical_positions.items():
                points = data['points']
                direction = data['direction']
                count_roi = data['count_roi']

                if is_inside_critical_location((centroid_x, centroid_y), points):
                    last_pos = last_positions[pos].get(obj_id)
                    
                    if last_pos is not None:
                        crossed = False
                        if direction == 'left_to_right':
                            crossed = last_pos[0] < count_roi <= centroid_x
                        elif direction == 'right_to_left':
                            crossed = last_pos[0] > count_roi >= centroid_x
                        elif direction == 'top_to_bottom':
                            crossed = last_pos[1] < count_roi <= centroid_y
                        elif direction == 'bottom_to_top':
                            crossed = last_pos[1] > count_roi >= centroid_y

                        ### to ensure uniqueness of counting i.e. same object won't be counted twice 
                        if crossed and obj_id not in counted_objects[pos]:
                            crossed_objects[pos] += 1
                            counted_objects[pos].add(obj_id)
                            blink_state[pos] = True
                            blink_frames[pos] = fps
                    
                    last_positions[pos][obj_id] = (centroid_x, centroid_y)

        # Update blink states for critical positions
        for pos in blink_state:
            if blink_state[pos]:
                blink_frames[pos] -= 1
                if blink_frames[pos] <= 0:
                    blink_state[pos] = False
        """
            Example:
        If an object crosses the ROI line for position1 at frame 100, and the video is 30 fps:

        At frame 100: blink_state['position1'] is set to True, blink_frames['position1'] is set to 30.
        Frames 101-129: The ROI line for position1 is drawn in white (blinking).
        Frame 130: blink_frames['position1'] reaches 0, blink_state['position1'] is set back to False.
        Frame 131 onwards: The ROI line for position1 is drawn in its normal color again.
        
        """

        # Display counts on the frame
        y_offset = 30
        for pos, count in crossed_objects.items():
            cv2.putText(frame, f"pos-{pos[-1]}/count: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 30

        # Draw counts near ROI lines for main output video
        draw_counts_near_roi(frame, critical_positions, crossed_objects, count_roi_colors)

        # Create visualizations
        speed_dots = create_speed_dots(frame.shape, tracked_objects, object_speeds)
        speed_area_visualization = visualize_speed_areas(frame.shape, tracked_objects, object_speeds, critical_positions, crossed_objects, count_roi_colors)
        
        # Draw blinking count ROI and counts on visualizations
        draw_blinking_count_roi(speed_area_visualization, blink_state)
        draw_blinking_count_roi(speed_dots, blink_state)
        draw_counts_near_roi(frame, critical_positions, crossed_objects, count_roi_colors)
        draw_counts_near_roi(speed_area_visualization, critical_positions, crossed_objects, count_roi_colors)
        draw_counts_near_roi(speed_dots, critical_positions, crossed_objects, count_roi_colors)

        # Create and save trail visualization if enabled
        if create_trail:
            canvas_copy = canvas.copy()
            plot_trail(canvas_copy, trails, tracked_objects, trail_colors, critical_positions, count_roi_colors, blink_state, crossed_objects)
            draw_blinking_count_roi(canvas_copy, blink_state)
            trail_out.write(canvas_copy)
            combo_frame = np.hstack((frame, canvas_copy))
            combo_out.write(combo_frame)

        # # Update grid counts for heatmap
        # for obj in tracked_objects:
        #     x1, y1, x2, y2, obj_id, _, cls, _ = obj
        #     grid_x = int((x1 + x2) / 2 / 40)
        #     grid_y = int((y1 + y2) / 2 / 40)
        #     if 0 <= grid_x < grid_counts.shape[1] and 0 <= grid_y < grid_counts.shape[0]:
        #         grid_counts[grid_y, grid_x] += 1

        # Write frames to output videos
        out.write(frame)
        speed_area_out.write(speed_area_visualization)
        speed_dots_out.write(speed_dots)
        
        # Create and write master analysis frame
        master_frame = combine_frames([frame, canvas_copy, speed_area_visualization, speed_dots])
        master_out.write(master_frame)


        # Display the frame (optional, for debugging)
        # Uncomment these lines if you need to view the processing in real-time
        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    # Cleanup and finalize video processing
    print(f"Total frames processed: {frame_count}")
    print(f"Saving videos...")

    cap.release()
    out.release()
    trail_out.release()
    speed_area_out.release()
    speed_dots_out.release()
    master_out.release()    
    combo_out.release()
    print("Videos saved")
    cv2.destroyAllWindows()
    print("Video processing completed")

if __name__ == "__main__":
    print("Processing started.")
    process_video(video_path, output_video_path, trail_video_path, speed_heatmap_path, master_analysis_path, create_trail)
    print("Processing completed.")