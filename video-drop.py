import cv2
import imutils
import os

def process_uploaded_video(video_path, output_path, initial_box):
    print(f"[INFO] Loading uploaded video: {video_path}")
    if not os.path.exists(video_path):
        print(f"[ERROR] Could not find video.")
        return

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: return

    first_frame = imutils.resize(first_frame, width=600)
    height, width = first_frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = cv2.TrackerCSRT_create()
    tracker.init(first_frame, initial_box)

    # --- PHASE 3 VARIABLES ---
    prev_cy = None             # To store the previous Center Y coordinate
    DROP_THRESHOLD = 15        # How many pixels per frame is considered a "drop"? (Tweak this!)
    drop_event_triggered = False 

    print("[INFO] Analyzing kinematics and searching for drops...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = imutils.resize(frame, width=600)
        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            
            # --- PHASE 3: KINEMATIC ANALYSIS (The Math) ---
            # Calculate the exact center point of the box
            cy = y + (h // 2) 
            
            # If we have a previous frame to compare to, calculate the speed (velocity)
            if prev_cy is not None:
                velocity_y = cy - prev_cy 
                
                # If the box moved down faster than our threshold, trigger the alert!
                if velocity_y > DROP_THRESHOLD:
                    drop_event_triggered = True
                    print(f"[ALERT] Drop detected! Downward velocity: {velocity_y} px/frame")

            # Update the previous Y coordinate for the next frame's math
            prev_cy = cy 
            # ----------------------------------------------

            # --- VISUAL ALERTS ---
            if drop_event_triggered:
                # Draw a RED box and big WARNING text if a drop happened
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, "WARNING: DROP DETECTED!", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            else:
                # Draw a normal GREEN box if everything is fine
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Status: Safe", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # If the product falls so fast it blurs completely, the tracker loses it.
            # This is also a good indicator of a severe drop!
            cv2.putText(frame, "TRACKING LOST (Possible Severe Impact)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[SUCCESS] Analysis complete. Saved as: {output_path}")

if __name__ == "__main__":
    input_video = "test_drop.mp4"   
    output_video = "tracked_result.mp4" 
    
    # Remember to adjust this box to fit your specific video!
    # (x, y, width, height)
    test_bounding_box = (200, 150, 200, 250) 
    
    process_uploaded_video(input_video, output_video, test_bounding_box)