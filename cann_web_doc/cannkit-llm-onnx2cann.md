# ONNX模型转换CANN模型

开发者需利用CANN提供的tools工具完成从ONNX模型到CANN模型的转换，模型转换完成后，即可开始集成。  

#### 配置CANN LLM模型NPU亲和适配文件

NPU亲和改造的脚本文件已默认在/CANN_LLM_Engine_Model/npu_tuned_export/npu_tuned_model下各个模型文件夹中，如果需要定制，请参见[CANN LLM模型NPU亲和适配说明](https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp/blob/master/CANN_LLM/CANN_LLM_Engine_Model/npu_tuned_export/npu_tuned_model/qwen2/README.md)。

开发者只需要按照下文模型转换流程执行对应的脚本，即可完成亲和化改造。  

#### 环境准备

模型转换需要使用tools/tools_omg/omg工具，继承量化环节的[DDK工具环境](./cannkit-llm-usage-environmental-preparation.md)。  

#### 模型转换

1. 修改CANN_LLM/CANN_LLM_Engine_Model/scripts_for_omc/to_omc.sh脚本，配置相应的omg_type、模型文件路径modelpath和量化文件路径compress_path，示例：

   ```
   omg_type=xxxxxx
   modelpath=./model.onnx
   compress_path=./quant_params_file
   outputpath=xxx  omgtoolpath=./tools/tools_omg/omg
   echo $PWDchmod 777 ./tools/tools_omg/omg export LD_LIBRARY_PATH=./tools/tools_omg/master/lib64
   if [ "$omg_type"=="Qwen25_1b5" ]; then
       echo ${omg_type} "trans start"
       ${omgtoolpath} --model ${modelpath} --framework 5 \
           --output ${outputpath} \
   --input_shape=
   "input_embed:1,-1,1536;attention_mask:1,1,-1,2048;position_ids:1,-1;past_key_in0:2048,2,1,128;past_value_in0:2048,2,1,128;past_key_in1:2048,2,1,128;past_value_in1:2048,2,1,128;past_key_in2:2048,2,1,128;past_value_in2:2048,2,1,128;past_key_in3:2048,2,1,128;past_value_in3:2048,2,1,128;past_key_in4:2048,2,1,128;past_value_in4:2048,2,1,128;past_key_in5:2048,2,1,128;past_value_in5:2048,2,1,128;past_key_in6:2048,2,1,128;past_value_in6:2048,2,1,128;past_key_in7:2048,2,1,128;past_value_in7:2048,2,1,128;past_key_in8:2048,2,1,128;past_value_in8:2048,2,1,128;past_key_in9:2048,2,1,128;past_value_in9:2048,2,1,128;past_key_in10:2048,2,1,128;past_value_in10:2048,2,1,128;past_key_in11:2048,2,1,128;past_value_in11:2048,2,1,128;past_key_in12:2048,2,1,128;past_value_in12:2048,2,1,128;past_key_in13:2048,2,1,128;past_value_in13:2048,2,1,128;past_key_in14:2048,2,1,128;past_value_in14:2048,2,1,128;past_key_in15:2048,2,1,128;past_value_in15:2048,2,1,128;past_key_in16:2048,2,1,128;past_value_in16:2048,2,1,128;past_key_in17:2048,2,1,128;past_value_in17:2048,2,1,128;past_key_in18:2048,2,1,128;past_value_in18:2048,2,1,128;past_key_in19:2048,2,1,128;past_value_in19:2048,2,1,128;past_key_in20:2048,2,1,128;past_value_in20:2048,2,1,128;past_key_in21:2048,2,1,128;past_value_in21:2048,2,1,128;past_key_in22:2048,2,1,128;past_value_in22:2048,2,1,128;past_key_in23:2048,2,1,128;past_value_in23:2048,2,1,128;past_key_in24:2048,2,1,128;past_value_in24:2048,2,1,128;past_key_in25:2048,2,1,128;past_value_in25:2048,2,1,128;past_key_in26:2048,2,1,128;past_value_in26:2048,2,1,128;past_key_in27:2048,2,1,128;past_value_in27:2048,2,1,128;new_kv_cache_pos:-1;embed_scales:1,-1,1" \
           --dynamic_dims="1,1,1,1,1;64,64,64,64,64" \
           --input_type=
   "past_key_in0:FP16;past_value_in0:FP16;past_key_in1:FP16;past_value_in1:FP16;past_key_in2:FP16;past_value_in2:FP16;past_key_in3:FP16;past_value_in3:FP16;past_key_in4:FP16;past_value_in4:FP16;past_key_in5:FP16;past_value_in5:FP16;past_key_in6:FP16;past_value_in6:FP16;past_key_in7:FP16;past_value_in7:FP16;past_key_in8:FP16;past_value_in8:FP16;past_key_in9:FP16;past_value_in9:FP16;past_key_in10:FP16;past_value_in10:FP16;past_key_in11:FP16;past_value_in11:FP16;past_key_in12:FP16;past_value_in12:FP16;past_key_in13:FP16;past_value_in13:FP16;past_key_in14:FP16;past_value_in14:FP16;past_key_in15:FP16;past_value_in15:FP16;past_key_in16:FP16;past_value_in16:FP16;past_key_in17:FP16;past_value_in17:FP16;past_key_in18:FP16;past_value_in18:FP16;past_key_in19:FP16;past_value_in19:FP16;past_key_in20:FP16;past_value_in20:FP16;past_key_in21:FP16;past_value_in21:FP16;past_key_in22:FP16;past_value_in22:FP16;past_key_in23:FP16;past_value_in23:FP16;past_key_in24:FP16;past_value_in24:FP16;past_key_in25:FP16;past_value_in25:FP16;past_key_in26:FP16;past_value_in26:FP16;past_key_in27:FP16;past_value_in27:FP16" \
           --output_type=
   "lm_logits:FP32;past_key0:FP16;past_value0:FP16;past_key1:FP16;past_value1:FP16;past_key2:FP16;past_value2:FP16;past_key3:FP16;past_value3:FP16;past_key4:FP16;past_value4:FP16;past_key5:FP16;past_value5:FP16;past_key6:FP16;past_value6:FP16;past_key7:FP16;past_value7:FP16;past_key8:FP16;past_value8:FP16;past_key9:FP16;past_value9:FP16;past_key10:FP16;past_value10:FP16;past_key11:FP16;past_value11:FP16;past_key12:FP16;past_value12:FP16;past_key13:FP16;past_value13:FP16;past_key14:FP16;past_value14:FP16;past_key15:FP16;past_value15:FP16;past_key16:FP16;past_value16:FP16;past_key17:FP16;past_value17:FP16;past_key18:FP16;past_value18:FP16;past_key19:FP16;past_value19:FP16;past_key20:FP16;past_value20:FP16;past_key21:FP16;past_value21:FP16;past_key22:FP16;past_value22:FP16;past_key23:FP16;past_value23:FP16;past_key24:FP16;past_value24:FP16;past_key25:FP16;past_value25:FP16;past_key26:FP16;past_value26:FP16;past_key27:FP16;past_value27:FP16"  \
           --compress_conf ${compress_path}  \
           --save_weights_as_external_data=true \
           --platform=xxxxxx \
           --target=omc
   ```

   脚本中的转换模型命令中的参数含义可参考[CANN Kit工具链文档说明](./cannkit-overall-parameter.md)。
2. 执行to_omc.sh脚本即可转换出对应的CANN格式的模型。

   脚本的输入/输出说明：

   输入：ONNX模型和量化配置文件。

   输出：CANN格式的模型。

模型转换完成后，即可开始集成。  
