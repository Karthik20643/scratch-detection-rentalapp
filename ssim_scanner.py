import cv2
import imutils
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def preprocess_image(image_path, width=600):
    image = cv2.imread(image_path)
    if image is None: return None, None
    resized_image = imutils.resize(image, width=width)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return resized_image, gray_image

def align_images(before_gray, after_gray, after_color):
    orb = cv2.ORB_create(nfeatures=5000)
    keypoints1, descriptors1 = orb.detectAndCompute(before_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(after_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    keep = int(len(matches) * 0.2)
    best_matches = matches[:keep]
    match_visual = cv2.drawMatches(before_gray, keypoints1, after_gray, keypoints2, best_matches, None)
    pts1 = np.zeros((len(best_matches), 2), dtype="float32")
    pts2 = np.zeros((len(best_matches), 2), dtype="float32")
    for i, match in enumerate(best_matches):
        pts1[i] = keypoints1[match.queryIdx].pt
        pts2[i] = keypoints2[match.trainIdx].pt
    matrix, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    height, width = before_gray.shape
    aligned_after_color = cv2.warpPerspective(after_color, matrix, (width, height))
    aligned_after_gray = cv2.cvtColor(aligned_after_color, cv2.COLOR_BGR2GRAY)
    return aligned_after_gray, aligned_after_color, match_visual

def compare_images(before_gray, aligned_after_gray):
    score, diff = ssim(before_gray, aligned_after_gray, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

def find_and_draw_damage(diff_image, aligned_after_color):
    _, thresh = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = aligned_after_color.copy()
    damage_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            damage_count += 1
    return output_image, thresh