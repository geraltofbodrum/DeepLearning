import pandas as pd

def analyze_lpips_scores(csv_file, model_name):

    print(f"Analyzing LPIPS scores for {model_name}...")

    df = pd.read_csv(csv_file)

    mean_score = df['LPIPS_Score'].mean()
    std_score = df['LPIPS_Score'].std()
    max_score = df['LPIPS_Score'].max()
    min_score = df['LPIPS_Score'].min()

    print(f"{model_name} LPIPS Score Analysis:")
    print(f"  Mean Score: {mean_score:.4f}")
    print(f"  Standard Deviation: {std_score:.4f}")
    print(f"  Maximum Score: {max_score:.4f}")
    print(f"  Minimum Score: {min_score:.4f}")

    return {
        "Model": model_name,
        "Mean Score": mean_score,
        "Standard Deviation": std_score,
        "Maximum Score": max_score,
        "Minimum Score": min_score
    }


if __name__ == "__main__":
    csv_files = {
        "SimSwap": "LPIPS_scores_same_identity_different_input_SimSwap.csv",
        "GHOST": "LPIPS_scores_same_identity_different_input_Ghost.csv",
        "Comparison": "LPIPS_scores_matching_Ghost_vs_SimSwap_same_identity.csv"
    }

    results = []
    for model_name, csv_file in csv_files.items():
        results.append(analyze_lpips_scores(csv_file, model_name))

    results_df = pd.DataFrame(results)

    # Save results to a CSV file
    results_df.to_csv("LPIPS_summary_statistics.csv", index=False)
