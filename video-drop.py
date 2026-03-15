import cv2
import imutils
import os

# --- IMPORT YOUR ORIGINAL CODE ---
# This assumes your first code is saved as 'ssim_scanner.py' in the same folder
# We are importing your specific functions!
from ssim_scanner import preprocess_image, align_images, compare_images, find_and_draw_damage

def process_and_inspect_video(video_path, output_video_path, initial_box):
    print(f"[INFO] Loading uploaded video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: return

    first_frame = imutils.resize(first_frame, width=600)
    height, width = first_frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    tracker = cv2.TrackerCSRT_create()
    tracker.init(first_frame, initial_box)

    # --- TRACKING & SETTLE VARIABLES ---
    prev_cy = None             
    DROP_THRESHOLD = 15        
    drop_detected = False 
    
    # New variables for taking the picture
    settle_counter = 0         
    frames_to_wait = 45 # Wait ~1.5 seconds (assuming 30fps) after the drop
    inspection_done = False

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = imutils.resize(frame, width=600)
        
        # Keep a clean copy of the frame BEFORE we draw green/red boxes on it
        clean_frame = frame.copy() 
        
        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cy = y + (h // 2) 
            
            # 1. Detect the Drop
            if prev_cy is not None and not drop_detected:
                velocity_y = cy - prev_cy 
                if velocity_y > DROP_THRESHOLD:
                    drop_detected = True
                    print(f"[ALERT] Drop detected! Waiting for product to settle...")

            prev_cy = cy 

            # 2. The Settle Timer & Photo Capture
            if drop_detected and not inspection_done:
                settle_counter += 1
                
                # Draw a warning on the video
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, "DROP EVENT - WAITING TO SETTLE", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Once the timer hits 45, take the picture!
                if settle_counter >= frames_to_wait:
                    print("[INFO] Product settled. Taking damage inspection photo...")
                    
                    # Save the clean (no boxes) frame as our after image
                    cv2.imwrite("captured_after.jpg", clean_frame)
                    inspection_done = True
                    
                    # RUN THE SSIM INSPECTION!
                    run_final_qa_check()
            
            elif not drop_detected:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Status: Safe", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[SUCCESS] Video processing complete.")


def run_final_qa_check():
    """
    This function feeds the newly captured video frame into your ORIGINAL code.
    """
    print("\n--- INITIATING QA STRUCTURAL SCAN ---")
    
    # The reference image your server already has
    img1_path = "before.jpg" 
    # The image our video tracker just took
    img2_path = "captured_after.jpg" 

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
         print("[ERROR] Missing images for comparison.")
         return

    # Call your original functions!
    color_before, gray_before = preprocess_image(img1_path)
    color_after, gray_after = preprocess_image(img2_path)

    aligned_gray, aligned_color, _ = align_images(gray_before, gray_after, color_after)
    score, diff_image = compare_images(gray_before, aligned_gray)
    final_output, mask = find_and_draw_damage(diff_image, aligned_color)

    # Save the final damage report image for the website to display
    cv2.imwrite("final_damage_report.jpg", final_output)
    
    print(f"[FINAL REPORT] SSIM Score: {score * 100:.2f}%")
    if score < 0.98: # If similarity is less than 98%
        print("[FINAL REPORT] STATUS: FAILED. Structural damage detected.")
    else:
        print("[FINAL REPORT] STATUS: PASSED. No significant damage found.")
    print("--------------------------------------\n")


if __name__ == "__main__":
    input_video = "test_drop.mp4"   
    output_video = "tracked_result.mp4" 
    test_bounding_box = (200, 150, 200, 250) 
    
    process_and_inspect_video(input_video, output_video, test_bounding_box)