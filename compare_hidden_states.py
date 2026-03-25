import os
import numpy as np
import matplotlib.pyplot as plt
import glob

def load_dumptxt(filepath):
    """
    Load hidden states from dumptxt format (one value per line).
    """
    try:
        return np.loadtxt(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_output_arm(filepath):
    """
    Load hidden states from outputARM format (space separated values).
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # If multiple lines, join them or just split entire content
            data = np.fromstring(content, sep=' ')
            return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_metrics(a, b):
    """
    Calculate Absolute Relative Error and Cosine Similarity.
    """
    if a.shape != b.shape:
        print(f"Shape mismatch: {a.shape} vs {b.shape}. Truncating to smaller size for comparison.")
        min_len = min(a.size, b.size)
        a = a[:min_len]
        b = b[:min_len]

    epsilon = 1e-6
    # Mean Absolute Relative Error
    are = np.mean(np.abs(a - b) / (np.abs(b) + epsilon))
    
    # Cosine Similarity
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        cos_sim = 0
    else:
        cos_sim = np.dot(a, b) / (norm_a * norm_b)
        
    return are, cos_sim

def plot_comparison(a, b, title, save_path):
    """
    Plot the first 100 elements of both vectors.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot first 100 elements (or less if size is small)
    limit = min(100, a.size, b.size)
    
    plt.plot(a[:limit], label='dumptxt (hidden_states)', alpha=0.7)
    plt.plot(b[:limit], label='outputARM', alpha=0.7, linestyle='--')
    
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def main():
    dump_dir = "/data/dahu/mlsys/MNN/dumptxt"
    output_arm_dir = "/data/dahu/mlsys/MNN/transformers/llm/export/model-2B/output"
    save_dir = "layer_figurex86"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Determine the number of hidden states (assuming 0 to 27 based on user description)
    # Or glob the dump_dir but user asked for mapping based on order.txt logic
    # Mapping Logic:
    # 0 -> _Add_output_0_0
    # 1 -> _Add_1_output_0_0
    # 2 -> _Add_2_output_0_0
    # i >= 3 -> _blocks.{i}_Add_1_output_0_0
    
    # Check max index from hidden_states_*.txt
    hidden_files = glob.glob(os.path.join(dump_dir, "hidden_states_*.txt"))
    indices = []
    for f in hidden_files:
        try:
            name = os.path.basename(f)
            idx = int(name.replace("hidden_states_", "").replace(".txt", ""))
            indices.append(idx)
        except ValueError:
            continue
            
    indices.sort()
    
    for idx in indices:
        dump_file = os.path.join(dump_dir, f"hidden_states_{idx}.txt")
        
        # Determine outputARM file name
        if idx == 0:
            output_name = "_Add_output_0_0"
        elif idx == 1:
            output_name = "_Add_1_output_0_0"
        elif idx == 2:
            output_name = "_Add_2_output_0_0"
        else:
            output_name = f"_blocks.{idx}_Add_1_output_0_0"
            
        output_file = os.path.join(output_arm_dir, output_name)
        
        if not os.path.exists(output_file):
            print(f"Warning: Output file not found: {output_file} for index {idx}")
            continue
            
        print(f"Processing Layer {idx}...")
        print(f"  Dump: {os.path.basename(dump_file)}")
        print(f"  ARM:  {os.path.basename(output_file)}")
        
        data_dump = load_dumptxt(dump_file)
        data_arm = load_output_arm(output_file)
        
        if data_dump is None or data_arm is None:
            continue
            
        are, cos_sim = calculate_metrics(data_dump, data_arm)
        
        print(f"  ARE: {are:.6f}")
        print(f"  CosSim: {cos_sim:.6f}")
        
        title = f"Layer {idx}: ARE={are:.4f}, CosSim={cos_sim:.4f}"
        save_path = os.path.join(save_dir, f"layer_{idx}.png")
        
        plot_comparison(data_dump, data_arm, title, save_path)
        print(f"  Saved plot to {save_path}")
        print("-" * 30)

if __name__ == "__main__":
    main()
