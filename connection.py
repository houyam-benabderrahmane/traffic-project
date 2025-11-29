import cv2
import socket  # For ESP32 communication
from ultralytics import YOLO

# ESP32 IP address and Port 
ESP32_IP = "192.168.19.149"  # <<< Replace with your ESP32's IP
ESP32_PORT = 1234             # <<< Use the same port number on ESP32 server

# Create a socket connection to ESP32
esp32_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    esp32_socket.connect((ESP32_IP, ESP32_PORT))
    print(f"✅ Connected to ESP32 at {ESP32_IP}:{ESP32_PORT}")  
except Exception as e:
    print(f"❌ Error connecting to ESP32: {e}")
    exit()

# Load the trained YOLOv8 model
model = YOLO("best (1).pt")

# IP Webcam URL (replace with your actual phone IP and port)
ip_webcam_url = "http://192.168.19.49:8080/video"

# Open the MJPEG stream
cap = cv2.VideoCapture(ip_webcam_url)

if not cap.isOpened():
    print("❌ Error: Could not open MJPEG stream.")
    exit()

# Get class names from YOLO model
class_names = model.names

last_sent_sign = None

# Loop through the stream and process frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Perform YOLOv8 inference
    results = model.predict(frame, conf=0.5)  # Adjust confidence threshold if needed

    # Extract predictions
    detections = results[0].boxes

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        class_id = int(box.cls[0])

        class_name = class_names[class_id] if class_id in class_names else "Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label and confidence
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Send the detected sign to ESP32
        valid_signs = [
            "Stop", "Speed Limit 20", "Speed Limit 30", "Speed Limit 40", "Speed Limit 60", 
            "Speed Limit 70", "Speed Limit 80", "Speed Limit 90", "Speed Limit 100"
        ]

        if class_name in valid_signs and class_name != last_sent_sign:
            try:
                esp32_socket.sendall((class_name + "\n").encode())
                print(f"📡 Sent to ESP32: {class_name}")
                last_sent_sign = class_name
            except Exception as e:
                print(f"❌ Error sending data to ESP32: {e}")
                break

    # Display the frame with detections
    cv2.imshow("YOLOv8 Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
esp32_socket.close()  # Close the socket connection