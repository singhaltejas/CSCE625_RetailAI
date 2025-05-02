#!/usr/bin/env python3
# YOLOv8 Shelf Product Detection - Main Inference Script
# This script implements product comparison between before and after shelf images

import os
import argparse
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from ultralytics import YOLO
from pathlib import Path
import time

# Get the base path of the project
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Default paths
DEFAULT_MODEL_PATH = os.path.join(BASE_PATH, 'models', 'best_model.pt')
DEFAULT_TEST_DIR = os.path.join(BASE_PATH, 'data', 'test', 'images')
DEFAULT_OUTPUT_DIR = os.path.join(BASE_PATH, 'comparison_results')

# Function to compare products between before and after images
def compare_product_images(model_path, before_image_path, after_image_path, output_dir=DEFAULT_OUTPUT_DIR, conf_threshold=0.25):
    """
    Compare product counts between before and after images to identify sold items

    Args:
        model_path: Path to the trained YOLOv8 model
        before_image_path: Path to the 'before' image
        after_image_path: Path to the 'after' image
        output_dir: Directory to save outputs
        conf_threshold: Confidence threshold for detection

    Returns:
        Dictionary of product changes, annotated images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train.py first to generate the model.")
        return None, None, None, None, None

    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, None

    # Check if images exist
    if not os.path.exists(before_image_path):
        print(f"Error: Before image not found at {before_image_path}")
        return None, None, None, None, None

    if not os.path.exists(after_image_path):
        print(f"Error: After image not found at {after_image_path}")
        return None, None, None, None, None

    print(f"Processing 'before' image: {before_image_path}")
    # Run detection on before image
    before_results = model(before_image_path, conf=conf_threshold)[0]
    before_boxes = before_results.boxes
    before_class_ids = before_boxes.cls.cpu().numpy().astype(int)

    # Count products in before image
    before_counts = Counter()
    for cls_id in before_class_ids:
        class_name = model.names[cls_id]
        before_counts[class_name] += 1

    print(f"Processing 'after' image: {after_image_path}")
    # Run detection on after image
    after_results = model(after_image_path, conf=conf_threshold)[0]
    after_boxes = after_results.boxes
    after_class_ids = after_boxes.cls.cpu().numpy().astype(int)

    # Count products in after image
    after_counts = Counter()
    for cls_id in after_class_ids:
        class_name = model.names[cls_id]
        after_counts[class_name] += 1

    # Calculate the difference (before - after)
    # Positive values mean products were sold (fewer in 'after' image)
    all_classes = set(before_counts.keys()) | set(after_counts.keys())
    difference_counts = {}

    for class_name in all_classes:
        before_count = before_counts.get(class_name, 0)
        after_count = after_counts.get(class_name, 0)
        difference = before_count - after_count
        difference_counts[class_name] = difference

    # Create annotated images
    before_annotated = before_results.plot()
    after_annotated = after_results.plot()

    # Save the annotated images
    before_output_path = os.path.join(output_dir, f"{Path(before_image_path).stem}_detected.jpg")
    after_output_path = os.path.join(output_dir, f"{Path(after_image_path).stem}_detected.jpg")

    cv2.imwrite(before_output_path, before_annotated)
    cv2.imwrite(after_output_path, after_annotated)
    print(f"Annotated images saved to {output_dir}")

    # Save comparison results
    comparison_path = os.path.join(output_dir, 'product_comparison.txt')
    with open(comparison_path, 'w') as f:
        f.write("PRODUCT COUNT COMPARISON (BEFORE vs AFTER)\n")
        f.write("="*50 + "\n\n")

        # Sort differences by absolute value (most change first)
        sorted_differences = sorted(difference_counts.items(),
                                   key=lambda x: abs(x[1]), reverse=True)

        for i, (class_name, diff) in enumerate(sorted_differences, 1):
            before_count = before_counts.get(class_name, 0)
            after_count = after_counts.get(class_name, 0)

            status = "SOLD" if diff > 0 else "ADDED" if diff < 0 else "UNCHANGED"
            f.write(f"{i}. {class_name}: {before_count} → {after_count} ({abs(diff)} {status})\n")

    print(f"Comparison results saved to: {comparison_path}")

    return difference_counts, before_annotated, after_annotated, before_counts, after_counts

# Display individual image object counts
def display_image_objects(image_name, counts, output_dir=DEFAULT_OUTPUT_DIR):
    """Display object counts for an individual image"""
    if not counts:
        print(f"No objects detected in the {image_name} image.")
        return
        
    print(f"\n{image_name} IMAGE OBJECT COUNTS:")
    print("="*50)

    # Sort by count (descending)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    for i, (class_name, count) in enumerate(sorted_counts, 1):
        print(f"{i}. {class_name}: {count} objects")

    print(f"\nTOTAL OBJECTS: {sum(counts.values())}")
    print(f"TOTAL UNIQUE CLASSES: {len(counts)}")

    # Save to file
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{image_name.lower()}_counts.txt")
    with open(output_path, 'w') as f:
        f.write(f"{image_name} IMAGE OBJECT COUNTS\n")
        f.write("="*50 + "\n\n")

        for i, (class_name, count) in enumerate(sorted_counts, 1):
            f.write(f"{i}. {class_name}: {count} objects\n")

        f.write(f"\nTOTAL OBJECTS: {sum(counts.values())}\n")
        f.write(f"TOTAL UNIQUE CLASSES: {len(counts)}\n")

    # Create visualization for individual image
    if counts:
        plt.figure(figsize=(12, 8))

        # Show top 10 classes or all if fewer than 10
        top_n = min(10, len(sorted_counts))
        top_classes = [item[0] for item in sorted_counts[:top_n]]
        top_values = [item[1] for item in sorted_counts[:top_n]]

        bars = plt.bar(top_classes, top_values, color='dodgerblue')

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')

        plt.title(f'{image_name} Image - Top {top_n} Product Counts')
        plt.xlabel('Product Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'{image_name.lower()}_counts.png'))
        plt.close()
        print(f"{image_name} count visualization saved.")

# Display the top sold products
def display_top_sold_products(difference_counts, output_dir=DEFAULT_OUTPUT_DIR, top_n=5):
    """Display the top sold products"""
    # Identify sold products (positive difference)
    sold_products = {k: v for k, v in difference_counts.items() if v > 0}

    if not sold_products:
        print("\nNo products were sold between the two images")
        return

    # Sort by most sold
    top_sold = sorted(sold_products.items(), key=lambda x: x[1], reverse=True)

    # Display top N sold products
    print(f"\nTOP {min(top_n, len(top_sold))} SOLD PRODUCTS:")
    print("=" * 30)

    for i, (class_name, diff) in enumerate(top_sold[:min(top_n, len(top_sold))], 1):
        print(f"{i}. {class_name}: {diff} units sold")

    # Create and display chart of top sold products
    if sold_products:
        display_top_n = min(top_n, len(sold_products))
        display_top_sold = top_sold[:display_top_n]
        classes, counts = zip(*display_top_sold)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, counts, color='crimson')

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')

        plt.title('Top Sold Products')
        plt.xlabel('Product Class')
        plt.ylabel('Units Sold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'top_sold_chart.png'))
        plt.close()
        print(f"Top sold products chart saved to {output_dir}/top_sold_chart.png")

# Main function for product comparison
def analyze_product_changes(model_path, before_image_path, after_image_path, output_dir=DEFAULT_OUTPUT_DIR, conf_threshold=0.25):
    """Compare two shelf images and identify sold products"""
    start_time = time.time()
    
    print("="*50)
    print("SHELF PRODUCT COMPARISON ANALYSIS")
    print("="*50)
    print(f"Before image: {before_image_path}")
    print(f"After image: {after_image_path}")
    print(f"Using model: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Output directory: {output_dir}")
    print("-"*50)

    # Run comparison
    differences, before_annotated, after_annotated, before_counts, after_counts = compare_product_images(
        model_path,
        before_image_path,
        after_image_path,
        output_dir,
        conf_threshold
    )

    if differences is None:
        print("Comparison failed. Please check error messages above.")
        return None

    # Display individual image object counts
    display_image_objects("BEFORE", before_counts, output_dir)
    display_image_objects("AFTER", after_counts, output_dir)

    # Display top sold products
    display_top_sold_products(differences, output_dir)

    # Display the annotated images side by side
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(before_annotated, cv2.COLOR_BGR2RGB))
    plt.title('Before Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(after_annotated, cv2.COLOR_BGR2RGB))
    plt.title('After Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visual_comparison.png'))
    plt.close()
    print(f"Side-by-side comparison saved to {output_dir}/visual_comparison.png")

    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nAnalysis completed in {execution_time:.2f} seconds")
    print(f"All results saved to {output_dir}")

    return differences

# Function to get test images from the test folder
def get_test_images(test_dir=DEFAULT_TEST_DIR, num_images=2):
    """Get test images from the test folder"""
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return None, None
        
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(test_dir).glob(f'*{ext}')))
    
    # Sort to ensure consistent selection
    image_files = sorted(image_files)
    
    if len(image_files) < 2:
        print(f"Error: Need at least 2 images in {test_dir}, but found {len(image_files)}")
        return None, None
        
    # Select first two images
    before_image = str(image_files[0])
    after_image = str(image_files[1])
    
    return before_image, after_image

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Shelf Product Comparison Tool')
    
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to trained YOLO model (default: {DEFAULT_MODEL_PATH})')
    
    parser.add_argument('--before', type=str, default=None,
                        help='Path to before image (defaults to first image in test directory)')
    
    parser.add_argument('--after', type=str, default=None,
                        help='Path to after image (defaults to second image in test directory)')
    
    parser.add_argument('--test-dir', type=str, default=DEFAULT_TEST_DIR,
                        help=f'Path to test images directory (default: {DEFAULT_TEST_DIR})')
    
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Path to output directory (default: {DEFAULT_OUTPUT_DIR})')
    
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}")
        print("Please run train.py first to generate the model, or specify the correct model path with --model")
        exit(1)
    
    # Get before and after images
    before_image = args.before
    after_image = args.after
    
    # If not specified, get from test directory
    if before_image is None or after_image is None:
        print(f"No images specified, selecting from test directory: {args.test_dir}")
        before_image, after_image = get_test_images(args.test_dir)
        
        if before_image is None or after_image is None:
            print("Failed to get test images. Please check that the test directory exists and contains images.")
            exit(1)
            
        print(f"Selected before image: {before_image}")
        print(f"Selected after image: {after_image}")
    
    # Run the analysis
    analyze_product_changes(
        args.model,
        before_image,
        after_image,
        args.output,
        args.conf
    )