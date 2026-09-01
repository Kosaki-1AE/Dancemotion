# bone_vector/vep/shared_data.py
import queue

# Create a thread-safe queue for frames
# Max size can prevent memory issues if consumer is slow
frame_queue = queue.Queue(maxsize=10)

# Flag to signal when the producer (main.py) should stop
stop_processing_flag = False