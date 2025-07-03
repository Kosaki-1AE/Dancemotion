# vep/streamlit.py
import streamlit as st
import cv2
import numpy as np
import time
import os
import sys
import threading # For running the frame producer in a separate thread
import queue

# --- NO sys.path.append() NEEDED for sibling imports! ---
# Direct imports are now possible as all files are in the same 'vep' folder
from modules.face import FaceDetector
from modules.hands import HandDetector
from modules.pose import PoseDetector
from shared_data import frame_queue, stop_processing_flag
from frame_producer import run_frame_processing # Import the function directly to run in thread

# Initialize detectors globally (cached by Streamlit to avoid re-loading models)
@st.cache_resource
def get_detectors():
    # Pass only the filename to the detector constructors now
    # The detector classes handle finding the files in the same directory
    return FaceDetector('face_landmarker.task'), \
           HandDetector('hand_landmarker.task'), \
           PoseDetector('pose_landmarker_heavy.task')

face_detector, hand_detector, pose_detector = get_detectors()

# --- Global state for the producer thread ---
# Use st.session_state for Streamlit's state management, especially for button presses
# and maintaining state across reruns.
if 'producer_thread' not in st.session_state:
    st.session_state.producer_thread = None
if 'producer_running' not in st.session_state:
    st.session_state.producer_running = False

def start_producer():
    # Only start if not already running
    if not st.session_state.producer_running:
        global stop_processing_flag # Access the global flag
        stop_processing_flag = False # Ensure stop flag is reset
        
        # Create and start the thread that runs the frame processing logic
        st.session_state.producer_thread = threading.Thread(target=run_frame_processing)
        st.session_state.producer_thread.daemon = True # Allows program to exit even if thread is running
        st.session_state.producer_thread.start()
        st.session_state.producer_running = True
        print("Streamlit: Producer thread initiated.")
        # Force a rerun to update the UI immediately
        st.rerun()
    else:
        print("Streamlit: Producer is already running.")

def stop_producer():
    # Only stop if currently running
    if st.session_state.producer_running:
        global stop_processing_flag # Access the global flag
        stop_processing_flag = True # Signal producer to stop

        # Wait for the producer thread to finish gracefully
        if st.session_state.producer_thread and st.session_state.producer_thread.is_alive():
            st.session_state.producer_thread.join(timeout=5) # Wait max 5 seconds
            if st.session_state.producer_thread.is_alive():
                print("Streamlit: Warning: Producer thread did not terminate gracefully.")
        
        st.session_state.producer_running = False
        print("Streamlit: Producer stopped.")
        # Force a rerun to update the UI immediately (e.g., clear video feed)
        st.rerun()
    else:
        print("Streamlit: Producer is not running.")


# Sidebar
st.sidebar.title("Dance Performance Analysis System")
st.sidebar.subheader("Mode Selection")
mode = st.sidebar.radio("Choose Mode", ["Chat", "Realtime", "Video", "Image"])

st.sidebar.subheader("Model Selection")
models = st.sidebar.multiselect("Choose Models", ["Face", "Hands", "Pose"], default=["Face", "Hands", "Pose"])

st.sidebar.subheader("Upload Options")
uploaded_file = None
if mode in ["Video", "Image"]:
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["mp4", "jpg", "png"])

# Main Panel
st.title("Dance Performance Analysis System")

# Placeholder for video feed. st.empty() is ideal for dynamic content.
video_feed_placeholder = st.empty()

# --- Realtime Mode Logic ---
if mode == "Realtime":
    col1, col2 = st.columns(2)
    with col1:
        # Use a unique key for the button if it might appear multiple times or cause conflicts
        if st.button("Start Realtime Analysis 🚀", key="start_realtime_btn"):
            start_producer()
    with col2:
        if st.button("Stop Realtime Analysis 🛑", key="stop_realtime_btn"):
            stop_producer()

    st.subheader("Live Video Feed")
    live_feed_container = st.empty() # Create a specific container for the live feed

    st.subheader("Facial Expression(Blendshapes)")
    blendshapes_container = st.empty() #for blendshapes data
    # Create an empty placeholder for blendshapes. We'll write to it dynamically.
    blendshapes_display_area = st.empty()
    
    # This loop runs continuously to display frames if the producer is active
    if st.session_state.producer_running:
        while st.session_state.producer_running: # Loop as long as the producer is flagged as running
            try:
                # Get a frame from the queue with a short timeout
                frame_bytes,blendshapes_data = frame_queue.get(timeout=0.05) # Reduced timeout for faster checks
                # Decode the JPEG bytes back to an OpenCV image
                frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                # Convert to RGB for Streamlit display
                processed_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Update the image in the Streamlit placeholder
                live_feed_container.image(processed_rgb_frame, channels="RGB", use_container_width=True)
            
                # --- Display Blendshapes using st.progress ---
                if blendshapes_data:
                    # Collect all blendshape display elements into a list
                    blendshape_elements = []
                    for shape in blendshapes_data:
                        name = shape['name']
                        score = shape['score']
                        # Append a markdown string for the label and a progress bar
                        blendshape_elements.append(f"**{name}**: {score:.4f}")
                        blendshape_elements.append(f"<div style='margin-bottom: 8px;'></div>") # Small spacing
                        blendshape_elements.append(st.progress(score, text=None)) # Streamlit handles the bar and percentage text if needed
                        
                    # Clear the blendshapes_display_area and write all new elements
                    with blendshapes_display_area.container():
                        st.empty() # Clear previous content
                        for element in blendshape_elements:
                            if isinstance(element, str):
                                st.markdown(element, unsafe_allow_html=True) # For labels and spacing
                            else:
                                # This handles the st.progress object itself.
                                # Streamlit handles updates if it's the same object,
                                # but recreating them in a container ensures proper layout per frame.
                                pass # The progress bar is already handled by Streamlit.

                        # The cleanest way: Recreate the entire content of the container
                        # This avoids complex update logic for individual st.progress objects
                        # within a tight loop.
                        new_html_content = ""
                        for shape in blendshapes_data:
                            name = shape['name']
                            score = shape['score']
                            new_html_content += f"**{name}**: {score:.4f}<br>"
                            # Use st.progress directly inside the loop's context manager,
                            # but for a continuous update in an empty placeholder,
                            # it's usually better to build up markdown string or re-render
                            # the whole container.
                            # For simple progress bars, a direct st.progress is easiest.
                            
                            # Let's simplify the blendshape display loop for clarity and directness.
                            # The problem with st.progress in a loop like this is that it generates
                            # a new widget on each iteration, which can cause performance issues or
                            # unexpected behavior.
                            # A better way is to prepare the markdown or a list of widgets to display.

                        # Option A: Pure Markdown with custom bars (more control over look)
                        blendshapes_html = "<div>"
                        for shape in blendshapes_data:
                            name = shape['name']
                            score = shape['score']
                            bar_width_percent = score * 100
                            blendshapes_html += f"""
                            <div style="display: flex; align-items: center; margin-bottom: 5px; font-family: sans-serif;">
                                <span style="min-width: 150px; font-weight: bold; text-align: left;">{name}:</span>
                                <div style="flex-grow: 1; height: 18px; background-color: #eee; border-radius: 3px; overflow: hidden; margin-right: 10px;">
                                    <div style="width: {bar_width_percent}%; height: 100%; background-color: #4CAF50; text-align: right; color: white; line-height: 18px; font-size: 12px; padding-right: 5px; box-sizing: border-box; transition: width 0.1s ease-out;">
                                        {score:.4f}
                                    </div>
                                </div>
                                <span style="min-width: 50px; text-align: right;">{score:.4f}</span>
                            </div>
                            """
                        blendshapes_html += "</div>"
                        blendshapes_display_area.markdown(blendshapes_html, unsafe_allow_html=True)

                        # Option B: Streamlit native widgets (most reliable for layout)
                        # This requires rebuilding the components each time.
                        # with blendshapes_display_area.container():
                        #     st.empty() # Clear previous content in this container
                        #     for shape in blendshapes_data:
                        #         name = shape['name']
                        #         score = shape['score']
                        #         st.write(f"**{name}**: {score:.4f}")
                        #         st.progress(score)
                        #     # If you want numbers next to the bar, use st.columns
                        #     # for shape in blendshapes_data:
                        #     #     name = shape['name']
                        #     #     score = shape['score']
                        #     #     col_name, col_bar, col_score = st.columns([0.4, 0.5, 0.1])
                        #     #     col_name.write(f"**{name}**")
                        #     #     col_bar.progress(score)
                        #     #     col_score.write(f"{score:.4f}")

                else:
                    blendshapes_display_area.info("No face blendshapes detected.")
                # --- End Display Blendshapes ---
            
            except queue.Empty:
                # If no frame is in the queue, wait a very short time and try again
                time.sleep(0.005) 
                # This loop needs to be very fast to give the illusion of real-time.
                # However, Streamlit's refresh model means the whole script reruns on interaction.
                # The st.empty() allows continuous updates within one Streamlit run.
                # No st.rerun() here to avoid constant full script reruns, which would be inefficient.
            except Exception as e:
                st.error(f"Error displaying frame or blendshapes: {e}")
                stop_producer() # Attempt to stop producer on display error
                break # Exit the display loop on error
    else:
        live_feed_container.empty() # Ensure the video area is clear if not running
        blendshapes_container.empty()

elif mode == "Video":
    st.subheader("Video Analysis")
    if uploaded_file:
        video_feed_placeholder.video(uploaded_file)
        st.info("Video playback is shown above. Analysis results will appear below.")
        # Future work: Add logic here to process the uploaded video frame by frame
        # using your detectors and display summaries.
    else:
        st.info("Upload a video file to analyze.")
elif mode == "Image":
    st.subheader("Image Analysis")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1) # Read as BGR image
        
        # Convert to RGB for MediaPipe processing
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        output_image = image.copy()

        # Perform detections based on selected models
        if "Face" in models:
            face_result = face_detector.detect(mp_image)
            output_image = face_detector.draw(output_image, face_result)
        if "Hands" in models:
            hand_result = hand_detector.detect(mp_image)
            output_image = hand_detector.draw(output_image, hand_result)
        if "Pose" in models:
            pose_result = pose_detector.detect(mp_image)
            output_image = pose_detector.draw(output_image, pose_result)
            
        # Convert the processed image back to RGB for Streamlit display
        processed_rgb_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        video_feed_placeholder.image(processed_rgb_image, channels="RGB", use_column_width=True)
        st.success("Image analysis complete!")
    else:
        st.info("Upload an image file to analyze.")
else: # Chat Mode
    st.subheader("Chat Mode")
    st.info("Select a mode from the sidebar to begin analysis or chat.")

# Common sections (always visible or conditionally visible based on analysis results)
# These sections would typically populate *after* analysis is done or when data is available.
if mode != "Chat":
    st.subheader("Emotion and Gesture Summaries")
    st.text("Emotion Summary: [Placeholder for emotion summary]")
    st.text("Gesture Summary: [Placeholder for gesture summary]")

    st.subheader("NLP-Generated Story")
    st.text("Story: [Placeholder for NLP-generated story]")

    st.subheader("Chat with AI")
    chat_input = st.text_input("Enter your message:")
    if st.button("Send", key="chat_send_btn"): # Added unique key
        st.text("AI Response: [Placeholder for AI response]")

    st.subheader("Suggested Dance Tactics")
    st.text("Dance Tactics: [Placeholder for suggested dance tactics]")
else:
    # Only show chat input in chat mode directly
    st.subheader("Chat with AI")
    chat_input = st.text_input("Enter your message:")
    if st.button("Send", key="chat_mode_send_btn"): # Added unique key
        st.text("AI Response: [Placeholder for AI response]")
