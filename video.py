import cv2
import os

# Load the video
# video_path = '/home/quincy/Downloads/video1.mp4'
# cap = cv2.VideoCapture(video_path)

# # Create a directory to store the images
# output_dir = '/home/quincy/Downloads/'
# os.makedirs(output_dir, exist_ok=True)

# frame_count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Save each frame to the directory
#     if frame_count % 10 == 0:
#         cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", frame)
#     frame_count += 1
#     # frame_count += 1

# print(f"Total frames: {frame_count}")

# cap.release()

from PIL import Image

image = Image.open('/home/quincy/Downloads/input.jpg')
image.save('/home/quincy/Downloads/output.pdf', 'PDF', quality=100)
