import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast # Used for safely evaluating string-formatted lists


# 1. Load Datasets
def load_datasets(anno_df_path, pred_df_path):
    """
    Load the annotation and prediction datasets.
    """
    print("Loading Datasets...")
    try:
        # Load the manual annotations (ground truth)
        anno_df = pd.read_csv(anno_df_path, index_col=0)
        
        # Load the system-generated predictions
        pred_df = pd.read_csv(pred_df_path, index_col=0)
        
        print("Successfully loaded 'anno_df.csv' and 'pred_df.csv'.\n")
        return anno_df, pred_df

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the CSV files are in the correct directory.")
        return None, None

def analyze_ground_truth_data(anno_df):
    """
    Analyze the ground truth annotation data.

    """
    print("\n" + "="*50)
    print("Analysis of Ground Truth Data (anno_df)")
    print("="*50)
    
    print("\nShape:", anno_df.shape)

    print("\nBasic Information:")
    anno_df.info()
    
    print("\nMissing Values:")
    print(anno_df.isnull().sum())
    
    print("\nSummary Statistics:")
    print(anno_df.describe())

    print("\nUnique Labels and Their Counts:")
    label_counts = anno_df['label'].value_counts()
    print(label_counts)

    print("\nTop 10 Unique Images and Their Counts:")
    image_counts = anno_df['image_id'].value_counts().sort_values(ascending=False)
    print(image_counts.head(10))

    print("\nAnnotations per Image:")
    annotations_per_image = anno_df.groupby('image_id').size()
    print("\nSummary statistics for annotations per image:")
    print(annotations_per_image.describe())
    
    print("\nUnique Values per Column:")
    for col in ['image_id', 'id', 'defect_class_id']:
        print(f"{col}: {anno_df[col].nunique()} unique values")

    

def analyze_prediction_data(pred_df, show_plot=True):
    """
    Analyze the prediction data.

    """
    print("\n" + "="*50)
    print("Analysis of Prediction Data (pred_df)")
    print("="*50)
    
    print("\nShape:", pred_df.shape)

    print("\nBasic Information:")
    pred_df.info()
    
    print("\nMissing Values:")
    print(pred_df.isnull().sum())

    print("\nObject type summary statistics:")
    print(pred_df.describe(include=['O']))
    
    print("\nNumeric type summary statistics:")
    print(pred_df.describe())

    print("\nTop 10 Unique Images and Their Counts:")
    image_counts = pred_df['image_id'].value_counts().sort_values(ascending=False)
    print(image_counts.head(10))

    print("\nUnique Prediction Classes and Their Counts:")
    prediction_counts = pred_df['prediction_class'].value_counts().sort_values(ascending=False)
    print(prediction_counts)


    print("\nDistribution of Confidence Scores:")
    if show_plot:
        plt.figure(figsize=(12, 6))
        sns.histplot(pred_df['confidence'], bins=50, kde=True)
        plt.title('Distribution of Model Confidence Scores')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.show()
        print("The plot above shows the distribution of the model's confidence scores across all predictions.")
    

def compare_datasets(anno_df, pred_df, show_plot=True):
    """
    Compare annotation and prediction datasets.
    """
    print("\n" + "="*50)
    print("--- Comparative Analysis: Ground Truth vs. Predictions ---")
    print("="*50)

    # Get the number of unique images in each dataset
    anno_images = set(anno_df['image_id'].unique())
    pred_images = set(pred_df['image_id'].unique())

    print(f"\nNumber of unique images in Annotation data: {len(anno_images)}")
    print(f"Number of unique images in Prediction data: {len(pred_images)}")
    
    # Get the number of unique id in each dataset
    anno_id = set(anno_df['id'].unique())
    pred_polygon_id = set(pred_df['polygon_id'].unique())

    print(f"\nNumber of unique id in Annotation data: {len(anno_id)}")
    print(f"\nNumber of unique images in Prediction data: {len(pred_polygon_id)}")


    # Create a summary DataFrame to compare counts per image
    anno_counts = anno_df.groupby('image_id').size().rename('num_annotations')
    pred_counts = pred_df.groupby('image_id').size().rename('num_predictions')

    # Combine the counts into a single DataFrame
    comparison_df = pd.concat([anno_counts, pred_counts], axis=1).fillna(0).astype(int)
    comparison_df['difference'] = comparison_df['num_predictions'] - comparison_df['num_annotations']

    print("\n(a) Comparison of Annotation vs. Prediction Counts per Image (First 10):")
    print(comparison_df.head(10))

    print("\n(b) Summary statistics for the difference (Predictions - Annotations):")
    print(comparison_df['difference'].describe())

    # Visualize the difference
    if show_plot:
        plt.figure(figsize=(12, 6))
        sns.histplot(comparison_df['difference'], bins=50, kde=False)
        plt.title('Distribution of Difference (Predictions - Annotations) per Image')
        plt.xlabel('Difference (Num Predictions - Num Annotations)')
        plt.ylabel('Number of Images')
        plt.show()
        print("The plot above shows how often the model over-predicts (positive values) or under-predicts (negative values).")
        print("A value of 0 means the model predicted the exact same number of defects as the human expert for that image.")
    

