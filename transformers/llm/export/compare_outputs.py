import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_tensor(file_path):
    """
    读取张量保存的文件。
    根据 MNN 的 dump 格式，这里假设文件是纯文本的浮点数序列。
    如果您的文件是二进制格式 (.bin 或 .mdl)，请将其改为 np.fromfile(file_path, dtype=np.float32)
    """
    try:
        # 尝试作为文本文件读取，获取所有浮点数
        with open(file_path, 'r') as f:
            data = f.read().split()
            # 过滤掉可能的非数值类元数据，仅保留数值
            values = [float(x) for x in data if x.replace('.', '', 1).replace('-', '', 1).replace('e', '', 1).isdigit()]
        return np.array(values, dtype=np.float32)
    except Exception as e:
        # 如果是二进制文件，可以尝试 fallback
        try:
            return np.fromfile(file_path, dtype=np.float32)
        except Exception as e2:
            print(f"无法读取文件 {file_path}: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="比较 MNN 两次运行的算子输出误差")
    parser.add_argument('--dir1', type=str, required=True, help="第一次运行的输出文件夹路径")
    parser.add_argument('--dir2', type=str, required=True, help="第二次运行的输出文件夹路径")
    parser.add_argument('--order', type=str, default='order.txt', help="算子顺序文件 (order.txt) 的路径")
    args = parser.parse_args()

    if not os.path.exists(args.order):
        print(f"找不到 order.txt 文件: {args.order}")
        return

    # 读取算子顺序
    with open(args.order, 'r') as f:
        # 去除空行，并提取文件名
        tensor_names = [line.strip() for line in f.readlines() if line.strip()]

    # 为了适应 order.txt 中可能包含的 "output/" 前缀
    tensor_names = [name.replace("output/", "") if name.startswith("output/") else name for name in tensor_names]

    max_diffs = []
    rmse_errors = []
    cosine_sims = []
    valid_tensors = []
    max_diff_details = []

    print(f"开始比较 {args.dir1} 和 {args.dir2} 中的数据...")
    print(len(tensor_names), "张量")

    first_bad_plotted = False

    for i, name in enumerate(tensor_names):
        file1 = os.path.join(args.dir1, name)
        file2 = os.path.join(args.dir2, name)

        if not os.path.exists(file1) or not os.path.exists(file2):
            print(f"警告：找不到文件对 => {name}，已跳过。")
            continue

        tensor1 = read_tensor(file1)
        tensor2 = read_tensor(file2)

        if tensor1 is None or tensor2 is None:
            continue

        if tensor1.shape != tensor2.shape:
            # 如果形状不一样，以最小的长度对比
            min_len = min(tensor1.size, tensor2.size)
            tensor1 = tensor1[:min_len]
            tensor2 = tensor2[:min_len]

        diff = np.abs(tensor1 - tensor2)
        if diff.size > 0:
            max_idx = np.argmax(diff)
            max_diff = diff[max_idx]
            val1 = tensor1[max_idx]
            val2 = tensor2[max_idx]
        else:
            max_diff = 0.0
            val1 = 0.0
            val2 = 0.0
            max_idx = -1
        
        # 计算 RMSE (均方根误差)
        rmse = np.sqrt(np.mean(diff ** 2)) if diff.size > 0 else 0.0
        
        # 计算余弦相似度
        dot_product = np.dot(tensor1, tensor2)
        norm1 = np.linalg.norm(tensor1)
        norm2 = np.linalg.norm(tensor2)
        if norm1 == 0 or norm2 == 0:
            cosine_sim = 1.0 if norm1 == norm2 else 0.0
        else:
            cosine_sim = dot_product / (norm1 * norm2)

        max_diffs.append(max_diff)
        rmse_errors.append(rmse)
        cosine_sims.append(cosine_sim)
        valid_tensors.append(name)
        max_diff_details.append((val1, val2, max_idx))

        # 新增逻辑：记录并绘制第一次余弦相似度低于0.5的算子输入输出
        if cosine_sim < 0.5 and not first_bad_plotted:
            first_bad_plotted = True
            print(f"\n[!] 发现第一个余弦相似度低于 0.5 的张量: {name} (相似度: {cosine_sim:.6f})")
            
            # 向前寻找该算子前置收集的输入张量(连续的 'Input_' 前缀文件)
            op_inputs = []
            for j in range(i - 1, -1, -1):
                if tensor_names[j].startswith("Input_"):
                    op_inputs.insert(0, tensor_names[j])
                else:
                    break
            
            # 组织需要绘制的张量名称
            plot_tensors = op_inputs + [name] if not name.startswith("Input_") else [name]
            print(f"正在准备绘制 {len(plot_tensors)} 个相关张量 (包含其输入和它的输出)...")
            
            plot_dir = "python_plt"
            os.makedirs(plot_dir, exist_ok=True)
            
            num_plots = len(plot_tensors)
            # 每个 tensor 画四张图：
            # 1. 原始值全量对比
            # 2. 绝对误差全量
            # 3. 原始值局部对比 (index 700~900)
            # 4. 绝对误差局部 (index 700~900)
            fig, axes = plt.subplots(num_plots * 4, 1, figsize=(14, 4 * num_plots * 4))
            # 保证 axes 是二维结构以便统一处理
            if num_plots * 4 == 1:
                axes = np.array(axes)
            
            for idx, t_name in enumerate(plot_tensors):
                ax_val = axes[idx * 4]
                ax_diff = axes[idx * 4 + 1]
                ax_val_zoom = axes[idx * 4 + 2]
                ax_diff_zoom = axes[idx * 4 + 3]
                
                t1 = read_tensor(os.path.join(args.dir1, t_name))
                t2 = read_tensor(os.path.join(args.dir2, t_name))
                
                # 不再进行等距采样，绘制全量数据
                step = 1
                
                # 画数值对比 (全量)
                if t1 is not None:
                    ax_val.plot(t1[::step], label='Run 1', color='blue', alpha=0.7, linewidth=2.5)
                if t2 is not None:
                    ax_val.plot(t2[::step], label='Run 2', color='red', alpha=0.7, linestyle='--', linewidth=2.5)
                    
                title_suffix = ""
                ax_val.set_title(f"Values (Full): {t_name}{title_suffix}", fontsize=11)
                ax_val.legend(loc='upper right')
                ax_val.grid(True)
                
                # 画绝对误差 (全量)
                if t1 is not None and t2 is not None:
                    min_l = min(len(t1), len(t2))
                    absolute_diff = np.abs(t1[:min_l] - t2[:min_l])
                    ax_diff.plot(absolute_diff[::step], label='Absolute Error (|Run1 - Run2|)', color='darkorange', linewidth=2.0)
                    avg_diff = np.mean(absolute_diff)
                    max_diff_val = np.max(absolute_diff)
                    err_title = f"Abs Error (Full): {t_name} (Mean: {avg_diff:.6f}, Max: {max_diff_val:.6f})"
                else:
                    err_title = f"Abs Error (Full): {t_name} (Missing data)"
                    
                ax_diff.set_title(err_title, fontsize=10)
                ax_diff.legend(loc='upper right')
                ax_diff.grid(True)
                
                # 画数值对比 (局部 Zoom 700-900)
                zoom_start, zoom_end = 700, 900
                if t1 is not None:
                    safe_end1 = min(zoom_end, len(t1))
                    if safe_end1 > zoom_start:
                        ax_val_zoom.plot(range(zoom_start, safe_end1), t1[zoom_start:safe_end1], 
                                         label='Run 1', color='blue', alpha=0.7, linewidth=2.5, marker='o', markersize=3)
                if t2 is not None:
                    safe_end2 = min(zoom_end, len(t2))
                    if safe_end2 > zoom_start:
                        ax_val_zoom.plot(range(zoom_start, safe_end2), t2[zoom_start:safe_end2], 
                                         label='Run 2', color='red', alpha=0.7, linestyle='--', linewidth=2.5, marker='x', markersize=3)
                        
                ax_val_zoom.set_title(f"Values (Zoom {zoom_start}-{zoom_end}): {t_name}", fontsize=11)
                ax_val_zoom.legend(loc='upper right')
                ax_val_zoom.grid(True)
                
                # 画绝对误差 (局部 Zoom 700-900)
                if t1 is not None and t2 is not None:
                    safe_end_diff = min(zoom_end, len(t1), len(t2))
                    if safe_end_diff > zoom_start:
                        zoom_diff = np.abs(t1[zoom_start:safe_end_diff] - t2[zoom_start:safe_end_diff])
                        ax_diff_zoom.plot(range(zoom_start, safe_end_diff), zoom_diff, 
                                          label='Absolute Error', color='darkorange', linewidth=2.0, marker='s', markersize=3)
                        avg_zoom_diff = np.mean(zoom_diff)
                        max_zoom_diff = np.max(zoom_diff)
                        err_title_zoom = f"Abs Error (Zoom {zoom_start}-{zoom_end}): {t_name} (Mean: {avg_zoom_diff:.6f}, Max: {max_zoom_diff:.6f})"
                    else:
                        err_title_zoom = f"Abs Error (Zoom {zoom_start}-{zoom_end}): {t_name} (Not enough data elements)"
                else:
                    err_title_zoom = f"Abs Error (Zoom {zoom_start}-{zoom_end}): {t_name} (Missing data)"
                    
                ax_diff_zoom.set_title(err_title_zoom, fontsize=10)
                ax_diff_zoom.legend(loc='upper right')
                ax_diff_zoom.grid(True)
                
            plt.tight_layout()
            safe_name = name.replace('/', '_').replace('\\', '_')
            out_file = os.path.join(plot_dir, f"first_bad_op_{safe_name}.png")
            plt.savefig(out_file)
            plt.close()
            print(f"该算子的对比图已保存至: {out_file}\n")

    if not max_diffs:
        print("没有找到有效的张量文件进行比较，请检查目录和 order.txt。")
        return

    # 打印最终的统计
    print("--------------------------------------------------")
    print(f"总计比较了 {len(valid_tensors)} 个张量。")
    print(f"平均 Max Diff: {np.mean(max_diffs):.6f}")
    print(f"最大 Max Diff: {np.max(max_diffs):.6f}")
    print(f"平均 RMSE: {np.mean(rmse_errors):.6f}")
    print(f"平均余弦相似度: {np.mean(cosine_sims):.6f}")
    print("--------------------------------------------------")

    print("\n=========== 算子余弦相似度排序 (从小到大) ===========")
    sorted_sims = sorted(zip(valid_tensors, cosine_sims), key=lambda x: x[1])
    
    # 打印到终端并保存到 txt 文件
    with open("cosine_similarity_sort.txt", "w") as f:
        for idx, (t_name, sim) in enumerate(sorted_sims):
            line = f"{idx + 1:03d}. 相似度: {sim:.6f} | 算子: {t_name}"
            print(line)
            f.write(line + "\n")
            
    print(f"===================================================\n上述排序结果已保存至: cosine_similarity_sort.txt\n")

    print("\n=========== 算子最大误差排序 (从大到小) ===========")
    sorted_diffs = sorted(zip(valid_tensors, max_diffs, max_diff_details), key=lambda x: x[1], reverse=True)
    
    # 打印到终端并保存到 txt 文件 (前20个打印到终端，全部保存到文件)
    with open("max_diff_sort.txt", "w") as f:
        for idx, (t_name, m_diff, details) in enumerate(sorted_diffs):
            val1, val2, m_idx = details
            line = f"{idx + 1:03d}. 最大误差: {m_diff:.6f} | 算子: {t_name} | 第一组值: {val1:.8f}, 第二组值: {val2:.8f} (一维索引: {m_idx})"
            f.write(line + "\n")
            if idx < 10:
                print(line)
    print("...")            
    print(f"===================================================\n上述排序结果已保存至: max_diff_sort.txt\n")

    # 可视化误差
    plt.figure(figsize=(15, 10))

    # 1. 绝对最大误差 (Max Diff)
    plt.subplot(3, 1, 1)
    plt.plot(max_diffs, marker='o', markersize=3, linestyle='-', color='r')
    plt.title('Max Absolute Difference per Operator')
    plt.ylabel('Max Diff')
    plt.grid(True)
    
    # 2. 均方根误差 (RMSE)
    plt.subplot(3, 1, 2)
    plt.plot(rmse_errors, marker='s', markersize=3, linestyle='-', color='b')
    plt.title('RMSE Error per Operator')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    # 3. 余弦相似度
    plt.subplot(3, 1, 3)
    plt.plot(cosine_sims, marker='^', markersize=3, linestyle='-', color='g')
    plt.title('Cosine Similarity (closer to 1.0 is better)')
    plt.ylabel('Cosine Sim')
    plt.xlabel('Operator Execution Sequence')
    plt.grid(True)

    plt.tight_layout()
    
    output_png = 'error_analysis.png'
    plt.savefig(output_png)
    print(f"\n误差分析曲线图已保存至: {output_png}")
    plt.show()

if __name__ == '__main__':
    main()
