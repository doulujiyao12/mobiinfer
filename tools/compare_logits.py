import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_data(filepath):
    """Loads data from a text file where each line is a float."""
    try:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
        # Load data, assuming one number per line or space separated
        data = np.loadtxt(filepath)
        return data.flatten()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def compare_and_plot(file1, file2, output_img="comparison_result.png"):
    print(f"Loading {file1}...")
    data1 = load_data(file1)
    print(f"Loading {file2}...")
    data2 = load_data(file2)

    if data1.shape != data2.shape:
        print(f"Warning: Shapes contain different number of elements: {data1.shape} vs {data2.shape}")
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        print(f"Truncated both to {min_len} elements for comparison.")

    # Calculate errors
    diff = data1 - data2
    abs_diff = np.abs(diff)
    
    # Calculate relative error
    # Add epsilon to denominator to avoid division by zero
    epsilon = 1e-6
    # We treat data2 as the reference/ground truth
    denominator = np.abs(data2)
    # Mask out very small values in reference to avoid exploding relative error
    valid_mask = denominator > epsilon
    
    rel_diff = np.zeros_like(diff)
    rel_diff[valid_mask] = abs_diff[valid_mask] / denominator[valid_mask]
    
    # Statistics
    max_abs_err = np.max(abs_diff)
    mean_abs_err = np.mean(abs_diff)
    # Filter relative error for reporting to avoid outliers skewing 'max' too much if denominator was tiny
    max_rel_err = np.max(rel_diff)
    mean_rel_err = np.mean(rel_diff)
    
    # Cosine Similarity
    norm1 = np.linalg.norm(data1)
    norm2 = np.linalg.norm(data2)
    cosine_sim = np.dot(data1, data2) / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0

    print("-" * 50)
    print(f"Comparison Summary:")
    print(f"File 1 (Target): {os.path.basename(file1)}")
    print(f"File 2 (Ref):    {os.path.basename(file2)}")
    print("-" * 50)
    print(f"Max Absolute Error:  {max_abs_err:.6f}")
    print(f"Mean Absolute Error: {mean_abs_err:.6f}")
    print(f"Max Relative Error:  {max_rel_err:.6%}")
    print(f"Mean Relative Error: {mean_rel_err:.6%}")
    print(f"Cosine Similarity:   {cosine_sim:.6f}")
    print("-" * 50)

    # Plotting
    plt.figure(figsize=(16, 12))
    plt.suptitle(f'Error Analysis: {os.path.basename(file1)} vs {os.path.basename(file2)}', fontsize=16)

    # 1. Direct Comparison (Scatter)
    plt.subplot(2, 2, 1)
    # Downsample for scatter plot if too many points to speed up rendering
    step = max(1, len(data1) // 10000)
    plt.scatter(data2[::step], data1[::step], alpha=0.5, s=2, label='Sampled Data', color='blue')
    
    # Ideal line
    min_val = min(np.min(data1), np.min(data2))
    max_val = max(np.max(data1), np.max(data2))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    
    plt.xlabel('Reference Value (File 2)')
    plt.ylabel('Target Value (File 1)')
    plt.title('Scatter Plot: Target vs Reference')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Absolute Error Histogram
    plt.subplot(2, 2, 2)
    plt.hist(abs_diff, bins=50, log=True, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Absolute Errors (Log Scale)')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count (Log)')
    plt.grid(True, alpha=0.3)

    # 3. Curve Comparison (All elements)
    plt.subplot(2, 2, 3)
    # Plot all elements as requested
    plt.plot(data1, label='Target (File 1)', alpha=0.8, linewidth=0.5)
    plt.plot(data2, label='Reference (File 2)', alpha=0.6, linestyle='--', linewidth=0.5)
    plt.title(f'Waveform Comparison (All {len(data1)} elements)')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Error Distribution per Index (to see if errors are localized)
    plt.subplot(2, 2, 4)
    # Downsample
    step_plot = max(1, len(abs_diff) // 2000)
    plt.plot(abs_diff[::step_plot], color='salmon', linewidth=0.5, label='Abs Error')
    plt.title('Absolute Error over Index')
    plt.xlabel(f'Index (sampled every {step_plot})')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_img)
    print(f"Analysis plot saved to: {os.path.abspath(output_img)}")

    # Save detailed error report to text file
    error_report_file = os.path.splitext(output_img)[0] + "_error_details.txt"
    try:
        # Create a structured array for saving
        # Columns: Target Value, Ref Value, Abs Diff, Rel Diff
        report_data = np.column_stack((data1, data2, abs_diff, rel_diff))
        
        header = f"Index\tTarget(File1)\tRef(File2)\tAbs_Diff\tRel_Diff\n" + "-" * 60
        
        with open(error_report_file, 'w') as f:
            f.write(f"Comparison Report\n")
            f.write(f"File 1: {file1}\n")
            f.write(f"File 2: {file2}\n")
            f.write(f"Max Abs Error: {max_abs_err}\n")
            f.write(f"Mean Abs Error: {mean_abs_err}\n")
            f.write(f"Max Rel Error: {max_rel_err}\n")
            f.write(f"Mean Rel Error: {mean_rel_err}\n")
            f.write("-" * 60 + "\n")
            # f.write("Target\tRef\tAbsDiff\tRelDiff\n") # np.savetxt header is simpler
            
            # Use savetxt to append data properly
            # We iterate to write so we can include index easily or just dump data
            # For simplicity using numpy savetxt for the data part
            np.savetxt(f, report_data, fmt='%.6f', delimiter='\t', header="Target\tRef\tAbsDiff\tRelDiff")
            
        print(f"Detailed error report saved to: {os.path.abspath(error_report_file)}")
    except Exception as e:
        print(f"Failed to save text report: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_logits.py <file1_target> <file2_reference> [output_image_path]")
        sys.exit(1)

    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]
    out_img = sys.argv[3] if len(sys.argv) > 3 else "error_analysis.png"

    compare_and_plot(file_path1, file_path2, out_img)
