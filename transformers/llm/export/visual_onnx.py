import onnx
import numpy as np

# 加载 ONNX 模型
model_path = "model/onnx/visual.onnx"
model = onnx.load(model_path)

# 获取所有权重（initializers）
initializers = model.graph.initializer

print(f"Total number of weight tensors: {len(initializers)}\n")

for idx, init in enumerate(initializers):
    name = init.name
    shape = tuple(init.dims)
    
    # 正确获取 NumPy 数据类型（兼容新版本 ONNX）
    try:
        dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
    except AttributeError:
        # 兼容非常老的 ONNX 版本（不推荐）
        import onnx.mapping
        dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type]

    # 解析数据
    if init.raw_data:
        weight_array = np.frombuffer(init.raw_data, dtype=dtype).reshape(shape)
    else:
        # 处理非 raw_data 存储的小张量（如常量）
        if init.data_type == onnx.TensorProto.FLOAT:
            weight_array = np.array(init.float_data, dtype=np.float32).reshape(shape)
        elif init.data_type == onnx.TensorProto.INT64:
            weight_array = np.array(init.int64_data, dtype=np.int64).reshape(shape)
        elif init.data_type == onnx.TensorProto.INT32:
            weight_array = np.array(init.int32_data, dtype=np.int32).reshape(shape)
        else:
            weight_array = None

    print(f"[{idx}] Name: {name}")
    print(f"      Type: {dtype}")
    print(f"      Shape: {shape}")
    
    if weight_array is not None and weight_array.size > 0:
        preview = weight_array.flatten()[:5]
        print(f"      First few values: {preview}")
    else:
        print("      (No data or unsupported type)")
    print("-" * 60)