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

# Define paths
video_path = 'videos/2.mp4'  # Update this path
output_folder = 'output'
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
model = YOLOv10('best_v10l.pt').to(device)
model.conf = 0.75
tracker = OCSort()
print(model.names)

# Dictionary to keep track of object counts
crossed_objects = {f'position{i+1}': 0 for i in range(4)}
counted_objects = {f'position{i+1}': set() for i in range(4)}

# Colors for different classes before crossing the ROI
class_colors = {
    0: (0, 0, 255),  # Red
    1: (255, 0, 0),  # Blue
    2: (0, 255, 255),  # Yellow
    3: (255, 255, 0),  # Cyan
    4: (255, 0, 255),  # Magenta
    5: (0, 128, 255),  # Orange
    6: (128, 0, 128),  # Purple
    7: (255, 255, 255),  # White
}

# Colors for different count ROI lines
count_roi_colors = {
    'position1': (0, 255, 0),  # Green
    'position2': (255, 255, 0),  # Cyan
    'position3': (0, 255, 255),  # Yellow
    'position4': (255, 0, 255),  # Magenta
}

# Define critical positions (x, y) and thresholds for counting
critical_positions = {
    'position1': {
        'points': [(525, 570), (525, 725), (695, 570), (695, 725)],
        'direction': 'bottom_to_top'
    },
    'position2': {
        'points': [(711, 236), (711, 439), (843, 236), (843, 439)],
        'direction': 'left_to_right'
    },
    'position3': {
        'points': [(1073, 439), (1073, 610), (1290, 439), (1290, 610)],
        'direction': 'top_to_bottom'
    },
    'position4': {
        'points': [(846, 833), (846, 1003), (1010, 833), (1010, 1003)],
        'direction': 'right_to_left'
    }
}

# Function to calculate the count ROI based on critical points
def calculate_count_roi(points, direction):
    """
    This function calculates the count ROI based on critical points and direction
    - input: points (list of tuples), direction (string)
    - output: count_roi (int) representing the x or y coordinate of the ROI line
    """

    if direction in ['left_to_right', 'right_to_left']:
        return (points[0][0] + points[2][0]) // 2
    elif direction in ['top_to_bottom', 'bottom_to_top']:
        return (points[0][1] + points[1][1]) // 2
    return None

# Update critical_positions with count_roi
for pos, data in critical_positions.items():
    data['count_roi'] = calculate_count_roi(data['points'], data['direction'])

def detect_objects(frame):
    """
    This function runs detection onf the frame using YOLOv10 model
    - input: frame
    - output: detections in [[int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)]]
    """
    # Use the YOLO model to detect objects in the frame
    results = model(frame)
    
    detections = []
    for result in results:
        # Extract bounding box, confidence, and class for each detection
        for detection in result.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
    
    # Return detections as a numpy array, or an empty array if no detections
    return np.array(detections) if detections else np.empty((0, 6))

def track_objects(frame, detections):
    """
    This function updates object tracks using the OC-SORT tracker
    - input: frame (numpy array), detections (numpy array)
    - output: tracked objects
    """
    # Use the OC-SORT tracker to update object tracks
    return tracker.update(detections, frame)

def crosses_threshold(centroid, threshold, direction):
    """
    This function checks if an object has crossed a threshold based on its direction
    - input: centroid (tuple), threshold (int), direction (string)
    - output: boolean indicating if threshold is crossed
    """
    
    # Check if the object has crossed the threshold based on its direction
    if direction == 'left_to_right':
        return centroid[0] >= threshold
    elif direction == 'right_to_left':
        return centroid[0] <= threshold
    elif direction == 'top_to_bottom':
        return centroid[1] >= threshold
    elif direction == 'bottom_to_top':
        return centroid[1] <= threshold
    return False

def is_inside_critical_location(centroid, points):
    """
    This function checks if a point is inside a polygon defined by points
    - input: centroid (tuple), points (list of tuples)
    - output: boolean indicating if centroid is inside the polygon
    """
    # Create a polygon from the given points
    polygon = Polygon(points)
    # Create a point from the centroid
    point = Point(centroid)
    # Check if the point is inside the polygon
    return polygon.contains(point)

def draw_blinking_count_roi(frame, blink_state):
    """
    This function draws blinking ROI lines on the frame
    - input: frame (numpy array), blink_state (dict)
    - output: None (modifies frame in-place)
    """

    for pos, data in critical_positions.items():
        count_roi = data['count_roi']
        # Determine color based on blink state
        roi_color = (255, 255, 255) if blink_state[pos] else count_roi_colors[pos]
        points = data['points']
        
        # Draw the ROI line based on the direction
        if data['direction'] in ['left_to_right', 'right_to_left']:
            cv2.line(frame, (count_roi, points[0][1]), (count_roi, points[1][1]), roi_color, 4)
            label_pos = (count_roi, points[0][1] - 10)
        elif data['direction'] in ['top_to_bottom', 'bottom_to_top']:
            cv2.line(frame, (points[0][0], count_roi), (points[2][0], count_roi), roi_color, 4)
            label_pos = (points[0][0], count_roi - 10)
        
        # Add label for the position
        cv2.putText(frame, f"pos-{pos[-1]}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)

def draw_info_on_canvas(canvas, crossed_objects):
    """
    This function draws count information and ROI lines on a canvas
    - input: canvas (numpy array), crossed_objects (dict)
    - output: None (modifies canvas in-place)
    """

    y_offset = 30
    # Draw count information for each position
    for pos, count in crossed_objects.items():
        cv2.putText(canvas, f"pos-{pos[-1]}/count: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 30

    # Draw ROI lines for each position
    for pos, data in critical_positions.items():
        count_roi = data['count_roi']
        points = data['points']
        roi_color = count_roi_colors[pos]
        if data['direction'] in ['left_to_right', 'right_to_left']:
            cv2.line(canvas, (count_roi, points[0][1]), (count_roi, points[1][1]), roi_color, 4)
        elif data['direction'] in ['top_to_bottom', 'bottom_to_top']:
            cv2.line(canvas, (points[0][0], count_roi), (points[2][0], count_roi), roi_color, 4)

def get_unique_color():
    """
    This function generates a cycle of unique colors
    - input: None
    - output: generator yielding unique RGB colors
    """
    # Create a color map with 20 distinct colors
    colors = plt.cm.get_cmap('tab20', 20)
    color_cycle = cycle(colors(np.linspace(0, 1, 20)))
    for color in color_cycle:
        # Convert color to 8-bit RGB format
        yield tuple(int(c * 255) for c in color[:3])

color_generator = get_unique_color()

def plot_trail(canvas, trails, tracked_objects, trail_colors, critical_positions, count_roi_colors, blink_state, crossed_objects):
    """
    This function plots object trails and counts on a canvas
    - input: canvas (numpy array), trails (dict), tracked_objects (list), trail_colors (dict),
             critical_positions (dict), count_roi_colors (dict), blink_state (dict), crossed_objects (dict)
    - output: None (modifies canvas in-place)
    """

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, _, cls, _ = obj
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)

        # Initialize trail and color for new objects
        if obj_id not in trails:
            trails[obj_id] = []
            trail_colors[obj_id] = next(color_generator)

        # Add current position to trail
        trails[obj_id].append((centroid_x, centroid_y))
        
        # Draw trail
        for i in range(1, len(trails[obj_id])):
            cv2.line(canvas, trails[obj_id][i - 1], trails[obj_id][i], trail_colors[obj_id], 2)
    
    # Draw counts near ROI
    draw_counts_near_roi(canvas, critical_positions, crossed_objects, count_roi_colors)

def combine_frames(frames):
    """
    This function combines four frames into a 2x2 grid
    - input: frames (list of numpy arrays)
    - output: combined frame (numpy array)
    """
    # Get the dimensions of the first frame
    height, width = frames[0].shape[:2]
    
    # Resize all frames to match the first frame's dimensions
    resized_frames = [cv2.resize(frame, (width, height)) for frame in frames]
    
    # Combine frames into a 2x2 grid
    top_row = np.hstack((resized_frames[0], resized_frames[1]))
    bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
    return np.vstack((top_row, bottom_row))

def estimate_speed(prev_pos, curr_pos, fps, pixels_per_meter=10):
    """
    This function estimates the speed of an object based on its previous and current positions
    - input: prev_pos (tuple), curr_pos (tuple), fps (int), pixels_per_meter (int)
    - output: speed in km/h (float)
    """

    if prev_pos is None:
        return 0
    # Calculate Euclidean distance between current and previous positions
    distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
    # Convert distance from pixels to meters
    distance_meters = distance / pixels_per_meter
    # Calculate time between frames
    time_seconds = 1 / fps
    # Calculate speed in meters per second
    speed_mps = distance_meters / time_seconds
    # Convert speed to kilometers per hour
    speed_kmph = speed_mps * 3.6
    return speed_kmph

def create_heatmap(frame, heatmap_data, colormap=cv2.COLORMAP_JET):
    """
    This function creates a heatmap overlay on a frame
    - input: frame (numpy array), heatmap_data (list of tuples), colormap (cv2 colormap)
    - output: frame with heatmap overlay (numpy array)
    """

    # Create an empty heatmap with the same dimensions as the frame
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
    # Plot each data point on the heatmap
    for pos, value in heatmap_data:
        cv2.circle(heatmap, pos, 20, value, -1)
    # Normalize the heatmap values to 0-255 range
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    # Apply color map to the heatmap
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), colormap)
    # Blend the heatmap with the original frame
    return cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)


def draw_count_roi(frame, critical_positions, count_roi_colors, blink_state):
    """
    This function draws count ROI lines on a frame
    - input: frame (numpy array), critical_positions (dict), count_roi_colors (dict), blink_state (dict)
    - output: None (modifies frame in-place)
    """
    for pos, data in critical_positions.items():
        count_roi = data['count_roi']
        points = data['points']
        # Determine color based on blink state
        roi_color = (255, 255, 255) if blink_state[pos] else count_roi_colors[pos]
        # Draw ROI line based on direction
        if data['direction'] in ['left_to_right', 'right_to_left']:
            cv2.line(frame, (count_roi, points[0][1]), (count_roi, points[1][1]), roi_color, 4)
        elif data['direction'] in ['top_to_bottom', 'bottom_to_top']:
            cv2.line(frame, (points[0][0], count_roi), (points[2][0], count_roi), roi_color, 4)

def draw_count_roi_on_canvas(canvas, critical_positions, count_roi_colors):
    """
    This function draws count ROI lines on a canvas
    - input: canvas (numpy array), critical_positions (dict), count_roi_colors (dict)
    - output: None (modifies canvas in-place)
    """

    for pos, data in critical_positions.items():
        count_roi = data['count_roi']
        points = data['points']
        roi_color = count_roi_colors[pos]
        # Draw ROI line based on direction
        if data['direction'] in ['left_to_right', 'right_to_left']:
            cv2.line(canvas, (count_roi, points[0][1]), (count_roi, points[1][1]), roi_color, 2)
        elif data['direction'] in ['top_to_bottom', 'bottom_to_top']:
            cv2.line(canvas, (points[0][0], count_roi), (points[2][0], count_roi), roi_color, 2)

def create_speed_dots(frame_shape, tracked_objects, object_speeds):
    """
    This function creates a visualization of object speeds as colored dots
    - input: frame_shape (tuple), tracked_objects (list), object_speeds (dict)
    - output: canvas with speed dots (numpy array)
    """

    # Create an empty canvas
    canvas = np.zeros(frame_shape, dtype=np.uint8)
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, _, cls, _ = obj
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)
        speed = object_speeds.get(obj_id, {'speed': 0})['speed']
        # Map speed to color using jet colormap
        color = plt.cm.jet(speed / 100)[:3]  # Adjust max speed as needed
        color = tuple(int(c * 255) for c in color)
        # Draw speed dot
        cv2.circle(canvas, (centroid_x, centroid_y), 5, color, -1)
        # Add label with object class, ID, and speed
        cv2.putText(canvas, f"{model.names[int(cls)]}:{obj_id}/{speed:.1f}kmph", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas

def visualize_speed_areas(frame_shape, tracked_objects, object_speeds, critical_positions, crossed_objects, count_roi_colors):
    """
    This function creates a visualization of speed areas with a color-coded grid
    - input: frame_shape (tuple), tracked_objects (list), object_speeds (dict),
             critical_positions (dict), crossed_objects (dict), count_roi_colors (dict)
    - output: canvas with visualized speed areas and legend (numpy array)
    """

    # Create an empty canvas
    canvas = np.zeros(frame_shape, dtype=np.uint8)
    grid_size = 40
    grid_speeds = np.zeros((frame_shape[0] // grid_size, frame_shape[1] // grid_size), dtype=float)
    grid_counts = np.zeros((frame_shape[0] // grid_size, frame_shape[1] // grid_size), dtype=int)

    # Calculate average speed for each grid cell
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, _, cls, _ = obj
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)
        grid_y, grid_x = centroid_y // grid_size, centroid_x // grid_size
        speed = object_speeds.get(obj_id, {'speed': 0})['speed']
        
        if 0 <= grid_y < grid_speeds.shape[0] and 0 <= grid_x < grid_speeds.shape[1]:
            grid_speeds[grid_y, grid_x] += speed
            grid_counts[grid_y, grid_x] += 1

    # Define speed ranges and corresponding colors
    speed_ranges = [
        (0, 20, (0, 0, 255)),    # Red for very low speed (0-20 km/h)
        (20, 40, (0, 165, 255)), # Orange for low speed (20-40 km/h)
        (40, 60, (0, 255, 255)), # Yellow for moderate speed (40-60 km/h)
        (60, 80, (0, 255, 0)),   # Green for high speed (60-80 km/h)
        (80, float('inf'), (255, 0, 0))  # Blue for very high speed (80+ km/h)
    ]

    def get_color(speed):
        for low, high, color in speed_ranges:
            if low <= speed < high:
                return color
        return (255, 255, 255)  # White for any unexpected values

    # Draw colored areas based on average speed
    for y in range(0, frame_shape[0], grid_size):
        for x in range(0, frame_shape[1], grid_size):
            grid_y, grid_x = y // grid_size, x // grid_size
            count = grid_counts[grid_y, grid_x]
            if count > 0:
                avg_speed = grid_speeds[grid_y, grid_x] / count
                color = get_color(avg_speed)
                cv2.rectangle(canvas, (x, y), (x + grid_size, y + grid_size), color, -1)
                cv2.putText(canvas, f"{avg_speed:.1f}", (x + 5, y + grid_size - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw critical positions
    for pos, data in critical_positions.items():
        points = data['points']
        cv2.polylines(canvas, [np.array(points)], True, (0, 255, 0), 2)
        cv2.putText(canvas, f"pos-{pos[-1]}", points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw counts near ROI lines
    draw_counts_near_roi(canvas, critical_positions, crossed_objects, count_roi_colors)

    # Add legend
    legend_height = 30 * len(speed_ranges)
    legend = np.zeros((legend_height, 150, 3), dtype=np.uint8)
    for i, (low, high, color) in enumerate(speed_ranges):
        y_start = i * 30
        cv2.rectangle(legend, (0, y_start), (20, y_start + 20), color, -1)
        if high != float('inf'):
            cv2.putText(legend, f"{low}-{high} km/h", (25, y_start + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(legend, f"{low}+ km/h", (25, y_start + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Combine canvas and legend
    canvas_with_legend = np.vstack((canvas, np.zeros((legend_height, canvas.shape[1], 3), dtype=np.uint8)))
    canvas_with_legend[canvas.shape[0]:, :legend.shape[1]] = legend

    return canvas_with_legend

def draw_counts_near_roi(frame, critical_positions, crossed_objects, count_roi_colors):
    """
    This function draws object counts near ROI lines
    - input: frame (numpy array), critical_positions (dict), crossed_objects (dict), count_roi_colors (dict)
    - output: None (modifies frame in-place)
    """

    for pos, data in critical_positions.items():
        count_roi = data['count_roi']
        points = data['points']
        roi_color = count_roi_colors[pos]
        count = crossed_objects[pos]
        
        # Determine label position based on ROI direction
        if data['direction'] in ['left_to_right', 'right_to_left']:
            label_pos = (count_roi, points[0][1] - 10)
        elif data['direction'] in ['top_to_bottom', 'bottom_to_top']:
            label_pos = (points[0][0], count_roi - 10)
        
        # Draw count label near ROI
        cv2.putText(frame, f"pos-{pos[-1]}/count: {count}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)

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
    print("Starting main processing loop...")

    # Main video processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the current frame
        detections = detect_objects(frame)
        
        # Track detected objects
        tracked_objects = track_objects(frame, detections)

        # Draw blinking count ROI on the frame
        draw_blinking_count_roi(frame, blink_state)

        speed_heatmap_data = [] 
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

            speed_heatmap_data.append(((centroid_x, centroid_y), speed))

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

        # Update grid counts for heatmap
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id, _, cls, _ = obj
            grid_x = int((x1 + x2) / 2 / 40)
            grid_y = int((y1 + y2) / 2 / 40)
            if 0 <= grid_x < grid_counts.shape[1] and 0 <= grid_y < grid_counts.shape[0]:
                grid_counts[grid_y, grid_x] += 1

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
    print("Processing started")
    process_video(video_path, output_video_path, trail_video_path, speed_heatmap_path, master_analysis_path, create_trail)
    print("Processing completed.")