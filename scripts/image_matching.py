import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
from tqdm.notebook import tqdm # For a nice progress bar


try:
    # Load the manual annotations (ground truth)
    anno_df = pd.read_csv('../data/anno_df.csv', index_col=0)
    
    # Load the system-generated predictions
    pred_df = pd.read_csv('../data/pred_df.csv', index_col=0)
    print("Successfully reloaded datasets.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the CSV files are in the correct directory.")
    exit()
except ImportError:
    print("ERROR: The 'shapely' library is required. Please install it using 'pip install shapely'")
    exit()

# --- 1. Utility Function to Parse Polygon Strings ---
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


# --- 2. Function to Calculate Intersection over Union (IoU) ---
def calculate_iou(poly1_coords, poly2_coords):
    """
    Calculates the Intersection over Union (IoU) of two polygons.
    """
    poly1 = Polygon(poly1_coords)
    poly2 = Polygon(poly2_coords)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

# --- 3. REVISED Core Matching Logic ---
print("\nStarting Prediction-to-Annotation Matching")
IOU_THRESHOLD = 0.5
results = []

# Group data by image_id to process each image individually
anno_grouped = anno_df.groupby('image_id')
pred_grouped = pred_df.groupby('image_id')
all_image_ids = set(anno_df['image_id'].unique()) | set(pred_df['image_id'].unique())

for image_id in tqdm(all_image_ids, desc="Processing Images"):
    
    # Get annotations and predictions for the current image
    try:
        gt_polys = anno_grouped.get_group(image_id)
        gt_list = gt_polys['polygon'].tolist()
    except KeyError:
        gt_list = []

    try:
        pred_polys = pred_grouped.get_group(image_id)
        pred_list = pred_polys['polygon'].tolist()
        confidences = pred_polys['confidence'].tolist()
    except KeyError:
        pred_list = []

    num_gt = len(gt_list)
    num_pred = len(pred_list)

    # Keep track of which items have been matched
    gt_matches = np.zeros(num_gt, dtype=bool)
    pred_matches = np.zeros(num_pred, dtype=bool)
    
    # If there are no predictions or no ground truths, we can skip the matrix calculation
    if num_gt == 0 or num_pred == 0:
        pass # Leftover FPs and FNs will be handled later
    else:
        # Create an IoU Matrix: rows are predictions, columns are ground truths
        iou_matrix = np.zeros((num_pred, num_gt))
        for i in range(num_pred):
            for j in range(num_gt):
                iou_matrix[i, j] = calculate_iou(pred_list[i], gt_list[j])

        # Implement the "best-pair-centric" greedy matching
        while iou_matrix.max() > 0:
            # Find the index of the highest IoU value in the matrix
            pred_idx, gt_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            max_iou = iou_matrix[pred_idx, gt_idx]
            
            # If the best available match is below our threshold, stop matching
            if max_iou < IOU_THRESHOLD:
                break
            
            # This is a successful match (True Positive)
            gt_matches[gt_idx] = True
            pred_matches[pred_idx] = True
            
            results.append({
                'image_id': image_id,
                'confidence': confidences[pred_idx],
                'iou': max_iou,
                'match_type': 'TP'
            })
            
            # Remove this pair from consideration by zeroing out their row and column
            iou_matrix[pred_idx, :] = 0
            iou_matrix[:, gt_idx] = 0

    # Classify leftovers
    # Any prediction not matched is a False Positive
    for i in range(num_pred):
        if not pred_matches[i]:
            results.append({
                'image_id': image_id,
                'confidence': confidences[i],
                'iou': 0, # No suitable match found
                'match_type': 'FP'
            })

 