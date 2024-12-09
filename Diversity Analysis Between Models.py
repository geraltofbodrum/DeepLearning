import lpips
import cv2
import os
import pandas as pd

def compute_diversity(img1_path, img2_path, lpips_model, target_size=(256, 256)):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)

    img1_tensor = lpips.im2tensor(img1).cuda()
    img2_tensor = lpips.im2tensor(img2).cuda()

    return lpips_model(img1_tensor, img2_tensor).item()

def compute_scores_for_matching_images(folder1, folder2, lpips_model, output_csv, target_size=(256, 256)):

    images_folder1 = {img: os.path.join(folder1, img) for img in os.listdir(folder1) if img.endswith(('.jpg', '.png'))}
    images_folder2 = {img: os.path.join(folder2, img) for img in os.listdir(folder2) if img.endswith(('.jpg', '.png'))}

    matching_images = set(images_folder1.keys()).intersection(set(images_folder2.keys()))

    results = []

    for img_name in matching_images:
        try:
            score = compute_diversity(images_folder1[img_name], images_folder2[img_name], lpips_model, target_size)
            results.append({'Image': img_name, 'LPIPS_Score': score})
            print(f"Computed LPIPS for: {img_name} -> Score: {score}")
        except Exception as e:
            print(f"Error computing LPIPS for {img_name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":

    lpips_model = lpips.LPIPS(net="alex").cuda()  #Using the AlexNet backbone

    folder1 = "/home/aras/Desktop/University Folder/Deep Learning/SimOut/Same Identity"  # Replace with your first image directory
    folder2 = "/home/aras/Desktop/University Folder/Deep Learning/Ghost Out/part6"  # Replace with your second image directory
    output_csv = "LPIPS_scores_matching_Ghost_vs_SimSwap_same_identity.csv"

    compute_scores_for_matching_images(folder1, folder2, lpips_model, output_csv, target_size=(256, 256))
