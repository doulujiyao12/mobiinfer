# 原始模型导出为友好结构ONNX模型

在模型量化与NPU亲和性设计完成后，将原始模型转换为友好结构ONNX模型，提升后续CANN模型转换效率。

1. 进入目录/CANN_LLM_Engine_Model/npu_tuned_export编辑model_info_target.yaml文件，在该文件中填写导出文件（model_arch）与模型文件路径。

   ```
   model_arch: qwen2 #可选择qwen2/qwen3/zhipu
   # 原始模型路径
   hf_model_path: you hf_model absolute path
   ```

   配置export_model_single脚本中的info_path参数，指向上述配置好的model_info_target.yaml路径。

   ```
   # info_path内填写model_info_target.yaml实际路径
       info_path = "model_info_target.yaml"
       if len(sys.argv) > 1: info_path = sys.argv[1]
       import yaml
       with open(info_path, "r") as f:
           config = yaml.load(f, Loader=yaml.FullLoader)
       print(config)
       if config.get("qlibs", None) is not None: import sys; sys.path.insert(0, config["qlibs"])
       for seq_len in config["seq_len"]:
           export_llama(
               export_config = config
           )
   ```

2. 运行export_model_single脚本（以Qwen2为例：export_model_single_qwen2.py）。

   ```
   python export_model_single_qwen2.py
   ```

   执行成功后，在model_info_target.yaml中配置的output_dir路径下会生成ONNX模型文件及pb权重文件。
3. ONNX模型与[量化参数](./cannkit-llm-usage-environmental-preparation.md)结合，转换得到端侧部署的模型。

