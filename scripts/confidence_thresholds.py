import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Required Data
try:
    # We need the original annotations to get the total number of ground truth objects.
    anno_df = pd.read_csv('../data/anno_df.csv', index_col=0)
    matched_df = pd.read_csv('../data/matched_df.csv', index_col=0)
    print("Successfully loaded annotation data. 'matched_df' is ready.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure 'anno_df.csv' is in the correct directory.")
    exit()

# The total number of actual defects is the total number of rows in the annotation file.
# This is the denominator for calculating recall.
total_ground_truths = len(anno_df)

# Define the range of confidence thresholds to test.
thresholds = np.arange(0.01, 1.0, 0.01)

metrics = []

for threshold in thresholds:
    # Consider all predictions with confidence >= current threshold
    subset = matched_df[matched_df['confidence'] >= threshold]
    
    # Calculate True Positives (TP) and False Positives (FP) from this subset
    tp = len(subset[subset['match_type'] == 'TP'])
    fp = len(subset[subset['match_type'] == 'FP'])
    
    # False Negatives (FN) are all the ground truths that were NOT detected by a TP prediction
    # at or above this confidence level.
    fn = total_ground_truths - tp
    
    # Calculate Precision, Recall, and F1-Score
    # Precision: Of all the predictions we made, how many were correct?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall: Of all the actual defects, how many did we find?
    recall = tp / total_ground_truths if total_ground_truths > 0 else 0
    
    # F1-Score: The harmonic mean of Precision and Recall.
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn
    })

# Convert the results into a DataFrame for easy analysis
metrics_df = pd.DataFrame(metrics)

# Find the Optimal Threshold

# The optimal threshold is the one that maximizes the F1-Score.
optimal_idx = metrics_df['f1_score'].idxmax()
optimal_threshold_data = metrics_df.loc[optimal_idx]


# 3. Visualize the Results
def visualize_the_results():
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(metrics_df['threshold'], metrics_df['precision'], 'b-', label='Precision')
    ax.plot(metrics_df['threshold'], metrics_df['recall'], 'g-', label='Recall')
    ax.plot(metrics_df['threshold'], metrics_df['f1_score'], 'r-', lw=3, label='F1-Score')

    # Mark the optimal threshold
    optimal_threshold = optimal_threshold_data['threshold']
    optimal_f1 = optimal_threshold_data['f1_score']
    ax.axvline(x=optimal_threshold, color='k', linestyle='--', lw=2, 
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    ax.plot(optimal_threshold, optimal_f1, 'ko', markersize=10, label=f'Max F1-Score = {optimal_f1:.2f}')

    # Formatting the plot
    ax.set_title('Precision, Recall, and F1-Score vs. Confidence Threshold', fontsize=16)
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, loc='best')
    plt.show()

