import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
import numpy as np
from IK_DH import plot_pose

# Load YOLO model
model = YOLO('best_1123.pt')
className = ["Matang", "Agak Matang", "Belum Matang"]

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get camera intrinsics for converting 2D to 3D
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Get intrinsics of the depth sensor
color_stream = profile.get_stream(rs.stream.color)
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

# Define the translation matrix from camera to end-effector (example: camera is 100mm higher on the Z axis)
camToArm = np.array([
    [1, 0, 0, 0],      # No translation in X
    [0, 1, 0, 0],      # No translation in Y
    [0, 0, 1, 85],     # 85 mm translation on Z axis
    [0, 0, 0, 1]       # Homogeneous transformation matrix
])

try:
    while True:
        # Get frames from RealSense
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        colorImage = np.asanyarray(color_frame.get_data())
        depthImage = np.asanyarray(depth_frame.get_data())

        # Mirror the color image along the Y axis (horizontal flip)
        colorImage = cv2.flip(colorImage, 1)  # Horizontal flip

        # Perform inference with YOLO model
        results = model(colorImage)

        # Iterate over detected results
        for result in results:
            for box in result.boxes:
                # Get confidence score
                confidence = box.conf[0].item()

                # Get class index
                class_id = int(box.cls[0])

                # Check if the label is "Matang" and confidence is above 0.85
                if className[class_id] == "Matang" and confidence > 0.85:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Calculate bounding box center (in pixel coordinates)
                    bbox_center_x = (x1 + x2) // 2
                    bbox_center_y = (y1 + y2) // 2

                    # Get depth value (in meters) at the center of the bounding box
                    depth_value = depthImage[bbox_center_y, bbox_center_x] * depth_scale

                    # Only proceed if a valid depth value is found
                    if depth_value > 0:
                        # Convert 2D pixel coordinates + depth value into 3D real-world coordinates
                        point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [bbox_center_x, bbox_center_y], depth_value)

                        # Extract real-world coordinates (X, Y, Z) in meters
                        real_x, real_y, real_z = point_3d

                        # Convert coordinates to millimeters
                        real_x_mm = real_x * 1000
                        real_y_mm = real_y * 1000
                        real_z_mm = real_z * 1000

                        # If necessary, adjust for mirror effect on X axis
                        real_x_mm = -real_x_mm  # Reverse X axis if mirrored

                        # Object coordinates in camera's coordinate system
                        object_coords_camera = np.array([real_x_mm, real_y_mm, real_z_mm, 1])

                        # Transform the coordinates to the end-effector's reference frame
                        object_coords_end_effector = np.dot(camToArm, object_coords_camera)

                        # Extract the transformed coordinates
                        x_ee, y_ee, z_ee = object_coords_end_effector[:3]

                        print(f"Object coordinates in end-effector frame: X={x_ee:.2f}mm, Y={y_ee:.2f}mm, Z={z_ee:.2f}mm")

                    # Draw the bounding box
                    cv2.ellipse(colorImage, 
                                (bbox_center_x, bbox_center_y), 
                                ((x2 - x1) // 2, (y2 - y1) // 2), 
                                0, 0, 360, (0, 255, 0), 2)

                    # Draw center point
                    cv2.circle(colorImage, (bbox_center_x, bbox_center_y), 5, (255, 0, 0), -1)

                    # Draw label above the bounding box
                    label = f"Matang: {confidence:.2f}"
                    cv2.putText(colorImage, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the color image with drawn ellipses, centers, and labels
        cv2.imshow('Color Image', colorImage)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
