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
count_heatmap_path = os.path.join(output_folder, f"{input_file_name_no_ext}_count_heatmap.mp4")
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
    if direction in ['left_to_right', 'right_to_left']:
        return (points[0][0] + points[2][0]) // 2
    elif direction in ['top_to_bottom', 'bottom_to_top']:
        return (points[0][1] + points[1][1]) // 2
    return None

# Update critical_positions with count_roi
for pos, data in critical_positions.items():
    data['count_roi'] = calculate_count_roi(data['points'], data['direction'])

def detect_objects(frame):
    results = model(frame)
    detections = []
    for result in results:
        for detection in result.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
    return np.array(detections) if detections else np.empty((0, 6))

def track_objects(frame, detections):
    return tracker.update(detections, frame)

def crosses_threshold(centroid, threshold, direction):
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
    polygon = Polygon(points)
    point = Point(centroid)
    return polygon.contains(point)

def draw_blinking_count_roi(frame, blink_state):
    for pos, data in critical_positions.items():
        count_roi = data['count_roi']
        roi_color = (255, 255, 255) if blink_state[pos] else count_roi_colors[pos]
        points = data['points']
        if data['direction'] in ['left_to_right', 'right_to_left']:
            cv2.line(frame, (count_roi, points[0][1]), (count_roi, points[1][1]), roi_color, 4)
            label_pos = (count_roi, points[0][1] - 10)
        elif data['direction'] in ['top_to_bottom', 'bottom_to_top']:
            cv2.line(frame, (points[0][0], count_roi), (points[2][0], count_roi), roi_color, 4)
            label_pos = (points[0][0], count_roi - 10)
        cv2.putText(frame, f"pos-{pos[-1]}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)

def draw_info_on_canvas(canvas, crossed_objects):
    y_offset = 30
    for pos, count in crossed_objects.items():
        cv2.putText(canvas, f"pos-{pos[-1]}/count: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 30

    for pos, data in critical_positions.items():
        count_roi = data['count_roi']
        points = data['points']
        roi_color = count_roi_colors[pos]
        if data['direction'] in ['left_to_right', 'right_to_left']:
            cv2.line(canvas, (count_roi, points[0][1]), (count_roi, points[1][1]), roi_color, 4)
        elif data['direction'] in ['top_to_bottom', 'bottom_to_top']:
            cv2.line(canvas, (points[0][0], count_roi), (points[2][0], count_roi), roi_color, 4)

def get_unique_color():
    colors = plt.cm.get_cmap('tab20', 20)
    color_cycle = cycle(colors(np.linspace(0, 1, 20)))
    for color in color_cycle:
        yield tuple(int(c * 255) for c in color[:3])

color_generator = get_unique_color()

def plot_trail(canvas, trails, tracked_objects, trail_colors, critical_positions, count_roi_colors, blink_state, crossed_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, _, cls, _ = obj
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)

        if obj_id not in trails:
            trails[obj_id] = []
            trail_colors[obj_id] = next(color_generator)

        trails[obj_id].append((centroid_x, centroid_y))
        
        for i in range(1, len(trails[obj_id])):
            cv2.line(canvas, trails[obj_id][i - 1], trails[obj_id][i], trail_colors[obj_id], 2)
    # Use the new helper function instead of the existing code for drawing counts
    draw_counts_near_roi(canvas, critical_positions, crossed_objects, count_roi_colors)

def combine_frames(frames):
    # Get the dimensions of the first frame
    height, width = frames[0].shape[:2]
    
    # Resize all frames to match the first frame's dimensions
    resized_frames = [cv2.resize(frame, (width, height)) for frame in frames]
    
    top_row = np.hstack((resized_frames[0], resized_frames[1]))
    bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
    return np.vstack((top_row, bottom_row))

def estimate_speed(prev_pos, curr_pos, fps, pixels_per_meter=10):
    if prev_pos is None:
        return 0
    distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
    distance_meters = distance / pixels_per_meter
    time_seconds = 1 / fps
    speed_mps = distance_meters / time_seconds
    speed_kmph = speed_mps * 3.6
    return speed_kmph

def create_heatmap(frame, heatmap_data, colormap=cv2.COLORMAP_JET):
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
    for pos, value in heatmap_data:
        cv2.circle(heatmap, pos, 20, value, -1)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), colormap)
    return cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)


def draw_count_roi(frame, critical_positions, count_roi_colors, blink_state):
    for pos, data in critical_positions.items():
        count_roi = data['count_roi']
        points = data['points']
        roi_color = (255, 255, 255) if blink_state[pos] else count_roi_colors[pos]
        if data['direction'] in ['left_to_right', 'right_to_left']:
            cv2.line(frame, (count_roi, points[0][1]), (count_roi, points[1][1]), roi_color, 4)
        elif data['direction'] in ['top_to_bottom', 'bottom_to_top']:
            cv2.line(frame, (points[0][0], count_roi), (points[2][0], count_roi), roi_color, 4)


def draw_count_roi_on_canvas(canvas, critical_positions, count_roi_colors):
    for pos, data in critical_positions.items():
        count_roi = data['count_roi']
        points = data['points']
        roi_color = count_roi_colors[pos]
        if data['direction'] in ['left_to_right', 'right_to_left']:
            cv2.line(canvas, (count_roi, points[0][1]), (count_roi, points[1][1]), roi_color, 2)
        elif data['direction'] in ['top_to_bottom', 'bottom_to_top']:
            cv2.line(canvas, (points[0][0], count_roi), (points[2][0], count_roi), roi_color, 2)

def create_speed_dots(frame_shape, tracked_objects, object_speeds):
    canvas = np.zeros(frame_shape, dtype=np.uint8)
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, _, cls, _ = obj
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)
        speed = object_speeds.get(obj_id, {'speed': 0})['speed']
        color = plt.cm.jet(speed / 100)[:3]  # Adjust max speed as needed
        color = tuple(int(c * 255) for c in color)
        cv2.circle(canvas, (centroid_x, centroid_y), 5, color, -1)
        cv2.putText(canvas, f"{model.names[int(cls)]}:{obj_id}/{speed:.1f}kmph", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas
def create_speed_heatmap(frame_shape, tracked_objects, object_speeds, critical_positions, crossed_objects, count_roi_colors):
    canvas = np.zeros(frame_shape, dtype=np.uint8)
    grid_size = 40
    grid_speeds = np.zeros((frame_shape[0] // grid_size, frame_shape[1] // grid_size), dtype=float)
    grid_counts = np.zeros((frame_shape[0] // grid_size, frame_shape[1] // grid_size), dtype=int)

    # Calculate average speed for each grid
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, _, cls, _ = obj
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)
        grid_y, grid_x = centroid_y // grid_size, centroid_x // grid_size
        speed = object_speeds.get(obj_id, {'speed': 0})['speed']
        
        if 0 <= grid_y < grid_speeds.shape[0] and 0 <= grid_x < grid_speeds.shape[1]:
            grid_speeds[grid_y, grid_x] += speed
            grid_counts[grid_y, grid_x] += 1

    # Define color ranges with 10 km/h intervals
    max_speed = 100  # Adjust this value based on your expected maximum speed
    num_intervals = max_speed // 10
    color_ranges = []
    for i in range(num_intervals):
        low = i * 10
        high = (i + 1) * 10
        r = int(255 * i / (num_intervals - 1))
        b = int(255 * (num_intervals - 1 - i) / (num_intervals - 1))
        color_ranges.append((low, high, (b, 0, r)))
    color_ranges.append((max_speed, float('inf'), (0, 0, 255)))  # Red for speeds above max_speed

    def get_color(speed):
        for low, high, color in color_ranges:
            if low <= speed < high:
                return color
        return (0, 0, 255)  # Default to red for very high speeds

    # Calculate average speed and draw heatmap
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

    # Add color bar
    color_bar_width = 30
    color_bar = np.zeros((frame_shape[0], color_bar_width, 3), dtype=np.uint8)
    bar_height = frame_shape[0] // len(color_ranges)
    for i, (low, high, color) in enumerate(color_ranges):
        y_start = i * bar_height
        y_end = (i + 1) * bar_height
        cv2.rectangle(color_bar, (0, y_start), (color_bar_width, y_end), color, -1)
        cv2.putText(color_bar, f"{low}", (5, y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if high != float('inf'):
            cv2.putText(color_bar, f"{high}", (5, y_end - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(color_bar, f"{max_speed}+", (5, y_end - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    canvas = np.hstack((canvas, color_bar))

    return canvas

def draw_counts_near_roi(frame, critical_positions, crossed_objects, count_roi_colors):
    for pos, data in critical_positions.items():
        count_roi = data['count_roi']
        points = data['points']
        roi_color = count_roi_colors[pos]
        count = crossed_objects[pos]
        
        if data['direction'] in ['left_to_right', 'right_to_left']:
            label_pos = (count_roi, points[0][1] - 10)
        elif data['direction'] in ['top_to_bottom', 'bottom_to_top']:
            label_pos = (points[0][0], count_roi - 10)
        
        cv2.putText(frame, f"pos-{pos[-1]}/count: {count}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)
def process_video(video_path, output_video_path, trail_video_path, speed_heatmap_path, count_heatmap_path, master_analysis_path, create_trail):
    print("Starting video processing...")
    cap = cv2.VideoCapture(video_path)
    print(f"Video opened: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    trail_out = cv2.VideoWriter(trail_video_path, fourcc, fps, (frame_width, frame_height))
    # With these lines:
    speed_dots_path = os.path.join(output_folder, f"{input_file_name_no_ext}_speed_dots.mp4")
    speed_dots_out = cv2.VideoWriter(speed_dots_path, fourcc, fps, (frame_width, frame_height))
    speed_heatmap_out = cv2.VideoWriter(speed_heatmap_path, fourcc, fps, (frame_width, frame_height + 30))  # +30 for the color bar
    master_out = cv2.VideoWriter(master_analysis_path, fourcc, fps, (frame_width * 2, frame_height * 2))
    combo_video_path = os.path.join(output_folder, f"{input_file_name_no_ext}_trail_combo.mp4")
    combo_out = cv2.VideoWriter(combo_video_path, fourcc, fps, (frame_width * 2, frame_height))
    print("Video writers initialized")

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)
        tracked_objects = track_objects(frame, detections)

        draw_blinking_count_roi(frame, blink_state)

        speed_heatmap_data = []
        count_heatmap_data = []

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id, _, cls, _ = obj
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)

            color = class_colors.get(cls, (0, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (centroid_x, centroid_y), 3, color, -1)

            # Speed estimation
            if obj_id in object_speeds:
                speed = estimate_speed(object_speeds[obj_id]['prev_pos'], (centroid_x, centroid_y), fps)
                object_speeds[obj_id]['speed'] = speed
                object_speeds[obj_id]['prev_pos'] = (centroid_x, centroid_y)
            else:
                object_speeds[obj_id] = {'prev_pos': (centroid_x, centroid_y), 'speed': 0}

            speed = object_speeds[obj_id]['speed']
            cv2.putText(frame, f"{model.names[int(cls)]}:{obj_id}/{speed:.1f}kmph", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            speed_heatmap_data.append(((centroid_x, centroid_y), speed))

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

                        if crossed and obj_id not in counted_objects[pos]:
                            crossed_objects[pos] += 1
                            counted_objects[pos].add(obj_id)
                            blink_state[pos] = True
                            blink_frames[pos] = fps
                    
                    last_positions[pos][obj_id] = (centroid_x, centroid_y)
                    count_heatmap_data.append(((centroid_x, centroid_y), crossed_objects[pos]))

        # Update blink states
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


        # With these lines:
        speed_dots = create_speed_dots(frame.shape, tracked_objects, object_speeds)
        speed_heatmap = create_speed_heatmap(frame.shape, tracked_objects, object_speeds, critical_positions, crossed_objects, count_roi_colors)

        # Draw blinking count ROI on speed heatmap and count heatmap
        draw_blinking_count_roi(speed_heatmap, blink_state)
        draw_blinking_count_roi(speed_dots, blink_state)

        # Draw counts near ROI lines for main output video, speed heatmap, and count heatmap
        draw_counts_near_roi(frame, critical_positions, crossed_objects, count_roi_colors)
        draw_counts_near_roi(speed_heatmap, critical_positions, crossed_objects, count_roi_colors)
        draw_counts_near_roi(speed_dots, critical_positions, crossed_objects, count_roi_colors)

        # Draw trails
        if create_trail:
            canvas_copy = canvas.copy()
            plot_trail(canvas_copy, trails, tracked_objects, trail_colors, critical_positions, count_roi_colors, blink_state, crossed_objects)
            draw_blinking_count_roi(canvas_copy, blink_state)
            
            trail_out.write(canvas_copy)

            combo_frame = np.hstack((frame, canvas_copy))
            combo_out.write(combo_frame)
            

        # Update grid_counts
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id, _, cls, _ = obj
            grid_x = int((x1 + x2) / 2 / 40)
            grid_y = int((y1 + y2) / 2 / 40)
            if 0 <= grid_x < grid_counts.shape[1] and 0 <= grid_y < grid_counts.shape[0]:
                grid_counts[grid_y, grid_x] += 1

        # Write frames to output videos
        out.write(frame)
        speed_heatmap_out.write(speed_heatmap)
        speed_dots_out.write(speed_dots)
        
        # With this line:
        master_frame = combine_frames([frame, canvas_copy, speed_heatmap, speed_dots])
        master_out.write(master_frame)

        # Display the frame (optional, for debugging)
        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    # After the main loop
    print(f"Total frames processed: {frame_count}")
    print(f"Saving videos...")

    cap.release()
    out.release()
    trail_out.release()
    speed_heatmap_out.release()
    speed_dots_out.release()
    master_out.release()    
    combo_out.release()
    print("Videos saved")
    cv2.destroyAllWindows()
    print("Video processing completed")

if __name__ == "__main__":
    print("Script started")
    process_video(video_path, output_video_path, trail_video_path, speed_heatmap_path, count_heatmap_path, master_analysis_path, create_trail)
    print("Processing completed.")