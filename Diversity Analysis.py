import lpips
import cv2
import os
import pandas as pd
from itertools import combinations

def compute_diversity(img1_path, img2_path, lpips_model, target_size=(256, 256)):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)

    img1_tensor = lpips.im2tensor(img1).cuda()
    img2_tensor = lpips.im2tensor(img2).cuda()

    return lpips_model(img1_tensor, img2_tensor).item()

def compute_scores_for_all_pairs(image_dir, lpips_model, output_csv, target_size=(256, 256)):

    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png'))]
    results = []

    for img1_path, img2_path in combinations(image_paths, 2):
        try:
            score = compute_diversity(img1_path, img2_path, lpips_model, target_size)
            results.append({'Image1': os.path.basename(img1_path), 'Image2': os.path.basename(img2_path), 'LPIPS_Score': score})
            print(f"Computed LPIPS for: {img1_path} and {img2_path} -> Score: {score}")
        except Exception as e:
            print(f"Error computing LPIPS for {img1_path} and {img2_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    lpips_model = lpips.LPIPS(net="alex").cuda()  #Using the AlexNet backbone

    image_dir = "/home/aras/Desktop/University Folder/Deep Learning/Ghost Out/part8"  # Replace with your image directory
    output_csv = "LPIPS_scores_same_identity_different_input_Ghost.csv"

    compute_scores_for_all_pairs(image_dir, lpips_model, output_csv, target_size=(256, 256))
