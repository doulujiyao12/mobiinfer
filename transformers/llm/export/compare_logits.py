import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def read_logits(file_path):
    try:
        with open(file_path, 'r') as f:
            data = [float(line.strip()) for line in f if line.strip()]
        return np.array(data)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def compare_logits(file1, file2):
    print(f"Reading {file1}...")
    data1 = read_logits(file1)
    if data1 is None: return

    print(f"Reading {file2}...")
    data2 = read_logits(file2)
    if data2 is None: return

    if len(data1) != len(data2):
        print(f"Warning: Files have different lengths: {len(data1)} vs {len(data2)}")
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]

    # Absolute Error
    abs_diff = np.abs(data1 - data2)
    mean_abs_error = np.mean(abs_diff)
    max_abs_error = np.max(abs_diff)
    
    # Relative Error
    # Handling division by zero by adding epsilon or masking
    epsilon = 1e-8
    # Using the first file as the reference (ground truth)
    denominator = np.abs(data1)
    # Mask where denominator is very small to avoid huge relative errors on near-zero values
    valid_mask = denominator > epsilon
    
    if np.any(valid_mask):
        rel_error = np.zeros_like(data1)
        rel_error[valid_mask] = abs_diff[valid_mask] / denominator[valid_mask]
        mean_rel_error = np.mean(rel_error[valid_mask])
        max_rel_error = np.max(rel_error[valid_mask])
    else:
        mean_rel_error = 0.0
        max_rel_error = 0.0

    # Cosine Similarity
    norm1 = np.linalg.norm(data1)
    norm2 = np.linalg.norm(data2)
    if norm1 > 0 and norm2 > 0:
        cosine_sim = np.dot(data1, data2) / (norm1 * norm2)
    else:
        cosine_sim = 0.0

    print("-" * 40)
    print("Comparison Statistics:")
    print(f"Data Length: {len(data1)}")
    print(f"Mean Absolute Error (MAE): {mean_abs_error:.6f}")
    print(f"Max Absolute Error: {max_abs_error:.6f}")
    print(f"Mean Relative Error (MRE): {mean_rel_error:.6f} (excluding near-zero references)")
    print(f"Max Relative Error: {max_rel_error:.6f}")
    print(f"Cosine Similarity: {cosine_sim:.6f}")
    print("-" * 40)

    # Plotting
    try:
        plt.figure(figsize=(15, 15))
        
        # Subplot 1: Direct value comparison
        plt.subplot(3, 1, 1)
        # Using a subset if data is too large for clear visualization
        plot_len = min(len(data1), 1000) 
        if len(data1) > 1000:
             print(f"Plotting first {plot_len} points for clarity.")
        
        plt.plot(data1[:plot_len], label=os.path.basename(file1), alpha=0.7)
        plt.plot(data2[:plot_len], label=os.path.basename(file2), alpha=0.7, linestyle='--')
        plt.title(f'Logits Values Comparison (First {plot_len} points)')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Absolute Difference
        plt.subplot(3, 1, 2)
        plt.plot(abs_diff[:plot_len], color='red', label='Absolute Difference')
        plt.title(f'Absolute Difference (First {plot_len} points)')
        plt.legend()
        plt.grid(True)
        
        # Subplot 3: Around Max Error
        plt.subplot(3, 1, 3)
        max_idx = np.argmax(abs_diff)
        start_idx = max(0, max_idx - 250)
        end_idx = min(len(data1), max_idx + 250)
        
        x_range = range(start_idx, end_idx)
        plt.plot(x_range, data1[start_idx:end_idx], label=os.path.basename(file1), alpha=0.7)
        plt.plot(x_range, data2[start_idx:end_idx], label=os.path.basename(file2), alpha=0.7, linestyle='--')
        plt.axvline(x=max_idx, color='red', linestyle=':', label='Max Diff')
        plt.title(f'Values Around Max Absolute Difference (Index: {max_idx}, Diff: {abs_diff[max_idx]:.6f})')
        plt.legend()
        plt.grid(True)
        
        output_img = 'logits_comparison.png'
        plt.tight_layout()
        plt.savefig(output_img)
        print(f"Comparison plot saved to {os.path.abspath(output_img)}")
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == "__main__":
    # file1 = '/data/dahu/mlsys/MNN/transformers/llm/export/model-2B/test_data/logits.txt'
    file1 = '/data/dahu/mlsys/MNN/transformers/llm/export/model-2B/outputARM_logit/0_0.txt'
    file2 = '/data/dahu/mlsys/MNN/transformers/llm/export/logits.txt'
    
    if len(sys.argv) > 2:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    
    if not os.path.exists(file1):
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    if not os.path.exists(file2):
        print(f"Error: File not found: {file2}")
        sys.exit(1)
        
    compare_logits(file1, file2)
