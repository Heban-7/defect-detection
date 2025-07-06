import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
from tqdm.notebook import tqdm

# Load Data and Initial Setup
try:
    anno_df = pd.read_csv('../data/anno_df.csv', index_col=0)
    pred_df = pd.read_csv('../data/pred_df.csv', index_col=0)
    print("Successfully loaded datasets.")
except FileNotFoundError:
    print("ERROR: CSV files not found. Please ensure they are in the correct directory.")
    exit()
except ImportError:
    print("ERROR: The 'shapely' library is required. Please install it using 'pip install shapely'")
    exit()

# Utility Function to Parse Polygon Strings ---
# This function converts the string coordinates into a list of tuples.
def parse_polygon_string(poly_str):
    """
    Parses the string representation of polygon coordinates into a list of (x, y) tuples.
    Returns an empty list if the input is invalid.
    """
    if not isinstance(poly_str, str) or not poly_str:
        return []
    try:
        coords = [float(p) for p in poly_str.split(',')]
        # A valid polygon needs at least 3 points (6 coordinates)
        if len(coords) < 6:
            return []
        return list(zip(coords[::2], coords[1::2]))
    except (ValueError, TypeError):
        return []

# Apply the parsing function to create a clean 'polygon' column.
anno_df['polygon'] = anno_df['xy'].apply(parse_polygon_string)
pred_df['polygon'] = pred_df['xy'].apply(parse_polygon_string)

# Function to Calculate Intersection over Union (IoU)
def calculate_iou(poly1_coords, poly2_coords):
    """
    Calculates the Intersection over Union (IoU) of two polygons.
    
    Args:
        poly1_coords (list): A list of (x, y) tuples for the first polygon.
        poly2_coords (list): A list of (x, y) tuples for the second polygon.
        
    Returns:
        float: The IoU score, between 0.0 and 1.0.
    """
    # Create Shapely Polygon objects. The constructor is robust to invalid polygons.
    poly1 = Polygon(poly1_coords)
    poly2 = Polygon(poly2_coords)

    # Ensure polygons are valid for area calculation
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    # Calculate intersection and union areas
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    if union_area == 0:
        return 0.0  # Avoid division by zero

    return intersection_area / union_area

# --- 3. Core Matching Logic ---
IOU_THRESHOLD = 0.5 # A standard threshold for object detection tasks.
results = []
false_negatives_count = 0

# Get a list of all unique image_ids present in the prediction set
image_ids = pred_df['image_id'].unique()

# Use tqdm for a progress bar, as this can be a slow process
for image_id in tqdm(image_ids, desc="Processing Images"):
    
    # Get the annotations and predictions for the current image
    gt_polys = anno_df[anno_df['image_id'] == image_id]
    pred_polys = pred_df[pred_df['image_id'] == image_id]

    # If there are no ground truth annotations for this image, all predictions are False Positives
    if gt_polys.empty:
        for idx, pred in pred_polys.iterrows():
            results.append({
                'image_id': image_id,
                'confidence': pred['confidence'],
                'iou': 0,
                'match_type': 'FP' # False Positive
            })
        continue

    # If there are no predictions for this image, all ground truths are False Negatives
    if pred_polys.empty:
        false_negatives_count += len(gt_polys)
        continue

    # --- Create an IoU Matrix for the current image ---
    # Rows: predictions, Columns: ground truths
    iou_matrix = np.zeros((len(pred_polys), len(gt_polys)))
    for i, (_, pred) in enumerate(pred_polys.iterrows()):
        for j, (_, gt) in enumerate(gt_polys.iterrows()):
            iou_matrix[i, j] = calculate_iou(pred['polygon'], gt['polygon'])

    # --- Match predictions to ground truths using the matrix ---
    # We find the best match for each ground truth object to avoid multiple detections of the same object.
    
    # Keep track of which predictions and ground truths have been matched
    pred_matches = np.zeros(len(pred_polys), dtype=bool)
    gt_matches = np.zeros(len(gt_polys), dtype=bool)
    
    # Iterate through ground truths to find their best prediction match
    for j in range(len(gt_polys)):
        best_iou = 0
        best_pred_idx = -1
        
        # Find the best-matching, unmatched prediction for this ground truth
        for i in range(len(pred_polys)):
            if not pred_matches[i] and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_pred_idx = i

        # If a suitable match is found (IoU > threshold), mark it as a True Positive
        if best_pred_idx != -1 and best_iou >= IOU_THRESHOLD:
            pred_matches[best_pred_idx] = True
            gt_matches[j] = True
            
            pred = pred_polys.iloc[best_pred_idx]
            results.append({
                'image_id': image_id,
                'confidence': pred['confidence'],
                'iou': best_iou,
                'match_type': 'TP' # True Positive
            })

    # Predictions that were not matched to any ground truth are False Positives
    for i in range(len(pred_polys)):
        if not pred_matches[i]:
            pred = pred_polys.iloc[i]
            results.append({
                'image_id': image_id,
                'confidence': pred['confidence'],
                'iou': iou_matrix[i, :].max() if iou_matrix.shape[1] > 0 else 0, # Record the highest IoU it had
                'match_type': 'FP' # False Positive
            })
            
    # Ground truths that were not matched by any suitable prediction are False Negatives
    false_negatives_count += np.sum(~gt_matches)

