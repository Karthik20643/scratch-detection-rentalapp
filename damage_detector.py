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


# --- SPRINT TESTING BLOCK ---
# --- SPRINT TESTING BLOCK ---
if __name__ == "__main__":
    img1_path = "before.jpg"
    img2_path = "after.jpg"

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("[WARNING] Missing test images.")
    else:
        # Phase 1 & 2: Preprocess
        color_before, gray_before = preprocess_image(img1_path)
        color_after, gray_after = preprocess_image(img2_path)

        if gray_before is not None and gray_after is not None:
            # Phase 3: Align
            aligned_gray, aligned_color, match_visual = align_images(gray_before, gray_after, color_after)
            
            # Phase 4: Compare
            score, diff_image = compare_images(gray_before, aligned_gray)

            # Show the results!
            cv2.imshow("Original Before", gray_before)
            cv2.imshow("Aligned After", aligned_gray)
            cv2.imshow("Difference Map", diff_image)
            
            print("Press any key on the image windows to close them...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()