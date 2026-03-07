import cv2
import imutils
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def preprocess_image(image_path, width=600):
    """
    Loads an image, resizes it to a standard width, and converts it to grayscale.
    """
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"[ERROR] Could not load image at path: {image_path}")
        print("Please check if the file exists and the name is spelled correctly.")
        return None, None

    resized_image = imutils.resize(image, width=width)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    print(f"[SUCCESS] Processed: {image_path} | New Size: {gray_image.shape}")
    return resized_image, gray_image

def align_images(before_gray, after_gray, after_color):
    """
    Finds matching features between two images and warps the 'after' image
    to perfectly align with the 'before' image.
    """
    print("[INFO] Starting image alignment...")

    # 1. Initialize ORB
   # 1. Initialize ORB
    orb = cv2.ORB_create(nfeatures=5000)

    # 2. Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(before_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(after_gray, None)

    # 3. Match the features using Brute-Force Matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only the top 20% of matches
    keep = int(len(matches) * 0.2)
    best_matches = matches[:keep]

    # Draw the matches visually
    match_visual = cv2.drawMatches(before_gray, keypoints1, after_gray, keypoints2, best_matches, None)

    # 4. Extract (x, y) coordinates of the best matches
    pts1 = np.zeros((len(best_matches), 2), dtype="float32")
    pts2 = np.zeros((len(best_matches), 2), dtype="float32")

    for i, match in enumerate(best_matches):
        pts1[i] = keypoints1[match.queryIdx].pt
        pts2[i] = keypoints2[match.trainIdx].pt

    # 5. Calculate Homography Matrix
    matrix, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    # 6. Warp the 'After' image
    height, width = before_gray.shape
    aligned_after_color = cv2.warpPerspective(after_color, matrix, (width, height))
    aligned_after_gray = cv2.cvtColor(aligned_after_color, cv2.COLOR_BGR2GRAY)

    print("[SUCCESS] Images successfully aligned.")
    return aligned_after_gray, aligned_after_color, match_visual

def compare_images(before_gray, aligned_after_gray):
    """
    Compares the original image with the aligned returned image using SSIM
    to highlight structural differences (scratches/dents).
    """
    print("[INFO] Calculating Structural Similarity...")

    # Calculate SSIM
    # 'score' is a number between 0 and 1 (1 means they are identical)
    # 'diff' is a raw difference image matrix (floats between 0 and 1)
    score, diff = ssim(before_gray, aligned_after_gray, full=True)
    print(f"[RESULT] Image Similarity Score: {score * 100:.2f}%")

    # The diff image contains floating point numbers from 0 to 1.
    # OpenCV needs integers from 0 to 255 to display or process an image.
    diff = (diff * 255).astype("uint8")

    return score, diff

def find_and_draw_damage(diff_image, aligned_after_color):
    """
    Takes the raw difference map, filters out the noise, finds the 
    actual damage spots, and draws red bounding boxes around them.
    """
    print("[INFO] Highlighting damage...")

    # 1. Thresholding (The Filter)
    # SSIM makes identical areas WHITE and different areas DARK. 
    # We use THRESH_BINARY_INV to flip this: Damage becomes WHITE, everything else BLACK.
    # Otsu's method automatically calculates the best threshold value.
    _, thresh = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 2. Clean up the noise (Morphological Operations)
    # Sometimes dust or slight shadows appear as tiny white dots. We 'open' the image 
    # to erase tiny dots but keep the big scratches.
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 3. Find Contours (Trace the shapes)
    # This finds the outlines of all the white blobs on our black mask.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make a copy of the color image to draw on
    output_image = aligned_after_color.copy()
    damage_count = 0

    # 4. Loop through the contours and draw boxes
    for c in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(c)
        
        # FILTER: If the area is too small (e.g., under 50 pixels), ignore it. 
        # This prevents drawing boxes around specks of dust.
        if area > 50:
            # Get the coordinates for the bounding box
            x, y, w, h = cv2.boundingRect(c)
            
            # Draw a RED rectangle (BGR format: 0, 0, 255) with a thickness of 2
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            damage_count += 1

    print(f"[RESULT] Found {damage_count} distinct areas of damage.")
    return output_image, thresh
# --- SPRINT TESTING BLOCK ---
# --- SPRINT TESTING BLOCK ---
# --- SPRINT TESTING BLOCK ---
if __name__ == "__main__":
    img1_path = "before.jpg"
    img2_path = "after.jpg"

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("[WARNING] Missing test images.")
    else:
        color_before, gray_before = preprocess_image(img1_path)
        color_after, gray_after = preprocess_image(img2_path)

        if gray_before is not None and gray_after is not None:
            # Phase 3: Align
            aligned_gray, aligned_color, match_visual = align_images(gray_before, gray_after, color_after)
            
            # Phase 4: Compare
            score, diff_image = compare_images(gray_before, aligned_gray)

            # Phase 5 & 6: Find and Draw Damage
            final_output, mask = find_and_draw_damage(diff_image, aligned_color)

            # --- THE BIG REVEAL ---
            cv2.imshow("1. Original Before", color_before)
            cv2.imshow("2. The Mask (White = Damage)", mask)
            cv2.imshow("3. Final Result (Damage Highlighted)", final_output)
            
            print("Press any key on the image windows to close them...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()