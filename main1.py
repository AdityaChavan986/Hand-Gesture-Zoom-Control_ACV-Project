import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import numpy as np

# Set up Streamlit page configuration
st.set_page_config(page_title="Zoom GestureControl", layout="wide")

# Title of the Streamlit app
st.title("Hand Gesture Zoom Control")

# Sidebar for control settings
st.sidebar.header("Settings")

# Add an image uploader
uploaded_file = st.sidebar.file_uploader("Upload an Image for Overlay", type=["jpg", "jpeg", "png"])

# Start and Stop buttons
start_video = st.sidebar.button("Start Video")
stop_video = st.sidebar.button("Stop Video")

# Initialize session state variables to control the video stream
if "run_video" not in st.session_state:
    st.session_state.run_video = False

# Update session state based on button clicks
if start_video:
    st.session_state.run_video = True
elif stop_video:
    st.session_state.run_video = False

# Define the main function for hand gesture control
def run_hand_tracking(overlay_image):
    cap = cv2.VideoCapture(0)  # Try 0 for the default camera
    if not cap.isOpened():
        st.warning("Error: Could not access the camera.")
        return

    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height

    detector = HandDetector(detectionCon=0.7)
    startDist = None
    scale = 0
    cx, cy = 500, 500

    # Streamlit video display
    video_display = st.empty()

    while st.session_state.run_video:
        success, img = cap.read()
        if not success:
            st.warning("Failed to capture image.")
            break

        # Detect hands
        hands, img = detector.findHands(img)

        # Check if two hands are detected for the zoom gesture
        if len(hands) == 2:
            if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
                    detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:

                # Get landmarks and calculate distance between index tips
                lmList1 = hands[0]["lmList"]
                lmList2 = hands[1]["lmList"]

                if startDist is None:
                    length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                    startDist = length

                length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                scale = int((length - startDist) // 2)
                cx, cy = info[4:]

        else:
            startDist = None

        # Resize and overlay the uploaded image onto img with scaling
        try:
            h1, w1, _ = overlay_image.shape
            newH, newW = max(1, ((h1 + scale) // 2) * 2), max(1, ((w1 + scale) // 2) * 2)
            resized_overlay = cv2.resize(overlay_image, (newW, newH))

            # Ensure overlay coordinates are within bounds
            y1, y2 = max(0, cy - newH // 2), min(img.shape[0], cy + newH // 2)
            x1, x2 = max(0, cx - newW // 2), min(img.shape[1], cx + newW // 2)

            # Fit resized_overlay to the overlay region
            overlay_img = resized_overlay[0:y2-y1, 0:x2-x1]

            # Overlay resized image onto the main frame
            img[y1:y2, x1:x2] = overlay_img

        except Exception as e:
            st.error(f"Overlay error: {e}")

        # Display the resulting frame in Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Streamlit display
        video_display.image(img_rgb, channels="RGB")

    # Release resources
    cap.release()

# Run the main function if "Start Video" is pressed and an image is uploaded
if st.session_state.run_video:
    if uploaded_file:
        overlay_image = Image.open(uploaded_file)
        overlay_image = np.array(overlay_image)
        run_hand_tracking(overlay_image)
    else:
        st.warning("Please upload an image to use as an overlay.")
else:
    st.write("Click 'Start Video' to begin the hand tracking and zoom control.")
