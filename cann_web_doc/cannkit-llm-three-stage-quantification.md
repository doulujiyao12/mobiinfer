# 三段式量化步骤

量化工具采用三段式执行，所有阶段均需在GPU/CUDA环境下运行。  

#### 阶段一：权重量化（stage1）

1. 执行命令sh run.sh stage1，生成dopt_config.json配置文件。

   * 首次执行命令，会出现generate plugin quang config please set quant strategy firstly代表执行成功。此时工程会遍历模型的所有linear算子，在量化工程内会生成dopt_config.json文件。
2. 根据模型层类型，修改dopt_config.json文件内的quant_strategy值，配置量化策略。

   * decode层策略：Quant_act_weight_eco
   * lm_head层策略：Quant_lm_head
   * embedding层策略：Quant_Embed_MinMax

   推荐配置样例：（dopt_config.json） 。

   ```
   {
       "layer_stategy":{
           "model.embed_tokens": {
               "type": "<class 'torch.nn.modules.sparse.Embedding'>",
               "quant_strategy": "Quant_Embed_MinMax",
           },
           "model.layers.0.self_attn.q_proj": {
               "type": "<class 'torch.nn.modules.linear.Linear'>",
               "quant_strategy": "Quant_act_weight_eco",
               "weight" :{
                   "bit":4,
                   "group_size": 64
               },
               "input":{
                   "bit": 16
               }
           },
           "lm_head": {
               "type": "<class 'torch.nn.modules.linear.Linear'>",
               "quant_strategy": "Quant_lm_head"
           }
       }
   }
   ```

3. 按下列的示例再次修改dopt_config.json文件的quant_strategy后，再次执行sh run.sh stage1，即可进行权重量化。

   ```
   {
       "layer_stategy":{
           "model.embed_tokens": {
               "type": "<class 'torch.nn.modules.sparse.Embedding'>",
               "quant_strategy": "Quant_Embed_MinMax",
           },
           "model.layers.0.self_attn.q_proj": {
               "type": "<class 'torch.nn.modules.linear.Linear'>",
               "quant_strategy": "Quant_act_weight_eco",
               "weight" :{
                   "bit":4,
                   "group_size": 64
               },
               "input":{
                   "bit": 16
               },
               "output" : {
                   "bit"           : 16,
                   "per_channel"   : True,
                   "input_algo"    : "min_max",
               },
           },
           "lm_head": {
               "type": "<class 'torch.nn.modules.linear.Linear'>",
               "quant_strategy": "Quant_lm_head"
           }
       }
   }
   ```

* 出现"weight quant done!!!"日志，在量化工程内生成trained_quant_weight.pth文件。  

#### 阶段二：激活量化（stage2）

执行命令：sh run.sh stage2，如果日志输出"quant done!!!"，则说明成功生成量化文件trained.pth。  

#### 阶段三：量化参数提取（stage3）

执行命令：sh run.sh stage3，如果日志输出"build done!!!"，则说明成功生成文件。

生成文件及用途：

fake_quant_weight.pth是在导出ONNX模型时需要替换模型权重为该文件中的权重。

quant_params_file是模型的量化系数，用于后续omc转换。

embedding_weights+embedding_quant_scale文件是权重与量化系数，用于模型推理。  
