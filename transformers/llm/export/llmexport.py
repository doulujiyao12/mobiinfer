import os
import json
import glob
import warnings
import argparse

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import onnx
import torch

from utils.model import LlmModel, EmbeddingModel
from utils.tokenizer import LlmTokenizer
from utils.spinner import spinner_run
from utils.custom_op import FakeLinear
from utils.onnx_rebuilder import OnnxRebuilder
from utils.mnn_converter import MNNConverter
from utils.awq_quantizer import AwqQuantizer
from utils.smooth_quantizer import SmoothQuantizer
from utils.omni_quantizer import OmniQuantizer
from utils.torch_utils import onnx_export

class LlmExporter(torch.nn.Module):
    '''
    Base class for all llm model export. Inherits from [`torch.nn.Module`].
    '''
    def __init__(self, args):
        super().__init__()
        self.init_from_args(args)
        self.load_model(args.path)

    def init_from_args(self, args):
        self.args = args
        self.max_new_tokens = 1024
        self.dst_name = 'llm'
        # load config from args
        self.onnx_path = os.path.join(self.args.dst_path, 'onnx')
        if self.args.tokenizer_path is None:
            self.args.tokenizer_path = self.args.path
        if args.lm_quant_bit is None:
            self.args.lm_quant_bit = self.args.quant_bit
        if args.lm_quant_block is None:
            self.args.lm_quant_block = self.args.quant_block
        self.args.tie_word_embeddings = False
        # init export dst dir
        if not os.path.exists(self.args.dst_path):
            os.makedirs(self.args.dst_path)
        if not os.path.exists(self.onnx_path):
            os.makedirs(self.onnx_path)

    @spinner_run(f'load pretrained model ', True)
    def load_model(self, model_path):
        self.model = LlmModel.from_pretrained(model_path, args=self.args)
        self.tokenizer = LlmTokenizer.from_pretrained(
            self.args.tokenizer_path,
            model_type=self.model.config.model_type
        )
        self.model.tokenizer = self.tokenizer
        self.config = self.model.config
        self.model_type = self.config.model_type

        if self.args.awq or self.args.smooth:
            self.model.float()
        if self.args.export is not None:
            # set norm's weight as float for export
            def visit_module(module):
                if not isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
                    module.float()
                for name, child in module.named_children():
                    visit_module(child)
            visit_module(self.model)

        self.model_dynamic_axes = {
            "input_ids" : { 0: "seq_len" },
            "attention_mask" : { 2: "seq_len", 3: "seq_len" },
            "position_ids" : { 1: "seq_len" },
        }

        self.llm_config = {
            'model_type': self.config.model_type,
            'hidden_size' : self.config.hidden_size,
            'attention_mask': 'float', # Will be determined by model later
            'attention_type': self.config.attention_type,
            'is_mrope': self.model.rotary.is_mrope
        }
        self.llm_config.update(self.model.get_config())
        if self.config.sliding_window > 0:
            self.llm_config['sliding_window'] = self.config.sliding_window
        if hasattr(self.tokenizer, 'get_chat_template'):
             chat_template = self.tokenizer.get_chat_template()
             if chat_template is not None:
                 self.llm_config['jinja'] = {
                     'chat_template': chat_template
                 }
                 if self.tokenizer.bos_token:
                     self.llm_config['jinja']['bos'] = self.tokenizer.bos_token
                 if self.tokenizer.eos_token:
                     self.llm_config['jinja']['eos'] = self.tokenizer.eos_token
        # glm_ocr's HF template is too complex for minja parser, use simplified version
        if self.model_type == 'glm_ocr':
            self.llm_config['jinja'] = {
                'chat_template': "[gMASK]<sop>{% for message in messages %}{% if message.role == \"user\" %}<|user|>\n{{ message.content }}{% elif message.role == \"assistant\" %}<|assistant|>\n{{ message.content }}{% elif message.role == \"system\" %}<|system|>\n{{ message.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}",
                'eos': '<|endoftext|>'
            }

        # tie word embeddings
        self.args.tie_word_embeddings = not self.args.seperate_embed and self.model.lm.lm.weight.equal(self.model.embed.embed.weight)
        print(f'tie_word_embeddings: {self.args.tie_word_embeddings}')
        # Pass properties from model to exporter
        self.visual = self.model.visual
        self.audio = self.model.audio
        self.talker = self.model.talker
        self.mtp = self.model.mtp
        self.scale_emb = self.model.scale_emb

        return model_path

    @torch.no_grad()
    def response(self, query):
        # self.imitate_quant()
        self.model.decode_buffer = []
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        prompt = self.tokenizer.apply_chat_template(messages)
        if query not in prompt:
            prompt = query

        # Use model's tokenizer methods for encoding
        if self.model.visual is not None:
            input_ids = self.model.visual.str_to_ids(prompt)
        elif self.model.audio is not None:
            input_ids = self.model.audio.str_to_ids(prompt)
        else:
            input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['input_ids']

        seq_len = input_ids.numel()
        new_tokens = 0

        first_print = True
        
        while new_tokens < self.max_new_tokens:
            attention_mask = self.model.get_attention_mask(seq_len, new_tokens)
            position_ids = self.model.get_position_ids(seq_len, new_tokens, input_ids)
            input_embeds = self.model.embedding(input_ids)
            deepstack_embeds = self.model.visual.deepstacks() if self.model.visual is not None else None
            
            if first_print == True:
                first_print = False
                print(f'seq_len: {seq_len}, new_tokens: {new_tokens}')
                print(f'position_ids')
                print(position_ids.size())
                print(f'input_embeds')
                print(input_embeds.size())
                
            # print(f'seq_len: {seq_len}, new_tokens: {new_tokens}')
            # print(f'position_ids')
            # print(position_ids.size())
            # print(f'input_embeds')
            # print(input_embeds.size())
            # with open('debug.txt', 'w') as f:
            #     f.write(f'seq_len: {seq_len}, new_tokens: {new_tokens}\n')
            #     f.write(f'position_ids: {position_ids}\n')
            #     f.write(f'input_embeds: {input_embeds.size()}\n')
            #     f.write(f'input_ids_size: {input_ids.size()}\n')
            #     f.write(f'input_ids: {input_ids}\n')
            logits, _, _ = self.model.forward(
                input_ids=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                logits_index=torch.tensor([-1], dtype=torch.int32),
                deepstack_embeds=deepstack_embeds
            )

            token_id = torch.argmax(logits[:,-1,:])
            seq_len += 1
            new_tokens += 1
            if token_id in self.tokenizer.stop_ids:
                print("", end='\n')
                break

            # Use tokenizer's method for decoding
            word = self.tokenizer.id_to_str(token_id)
            print(word, end="", flush=True)
            input_ids = token_id
        print("seq_len: ", seq_len)

        if hasattr(self.model, 'talker') and self.model.talker is not None:
            self.model.talker.generate()

    def export_mtp(self):
        if self.mtp is None:
            return
        mtp_onnx = self.mtp.export(self.onnx_path)
        if self.mnn_converter:
            self.mtp.unloaded_ops['/lm/lm_head/Linear'] = self.unloaded_ops['/lm/lm_head/Linear']
            MNNConverter(self, self.mtp.unloaded_ops).export(mtp_onnx)

    def export_eagle(self):
        if self.args.eagle_path is None:
            return
        from utils.eagle import Eagle
        self.eagle = Eagle.get_eagle(self.model_type)(self.args.eagle_path, self.model)
        eagle_onnx, eagle_fc_onnx = self.eagle.export(self.onnx_path)
        if self.mnn_converter:
            MNNConverter(self, None).export(eagle_onnx)
            MNNConverter(self, None).export(eagle_fc_onnx)


    @spinner_run(f'export embedding to ')
    def export_embed(self):
        import ctypes
        from utils.torch_utils import quant as torch_quant

        if hasattr(self.model, 'word_embeddings'):
            # embedding model's embed
            tensor_data = self.model.word_embeddings.weight.data
        else:
            tensor_data = self.model.embed.embed.weight.data

        format_bit = getattr(self.args, 'embed_bit', 16)
        quant_block = getattr(self.args, 'quant_block', 64)
        symmetric = getattr(self.args, 'sym', False)

        if self.args.skip_weight:
            format_name = f'int{format_bit}' if format_bit < 16 else 'bf16'
            embedding_file = f'{self.args.dst_path}/embeddings_{format_name}.bin'
            # Calculate expected size
            if format_bit == 16:
                file_size = tensor_data.numel() * 2
            else:
                oc, ic = tensor_data.shape
                block_size = ic if quant_block == 0 else quant_block
                block_num = ic // block_size
                q_weight_size = (oc * ic * format_bit + 7) // 8
                alpha_size = oc * block_num * (1 if symmetric else 2) * 4
                file_size = q_weight_size + alpha_size
                self.llm_config['tie_embeddings'] = [0, q_weight_size, alpha_size, format_bit, quant_block]

            with open(embedding_file, 'wb') as f:
                if file_size > 0:
                    f.seek(file_size - 1)
                    f.write(b'\0')
            return embedding_file

        if format_bit == 16:
            # BF16 format
            tensor_data = tensor_data.bfloat16()
            data_ptr = tensor_data.untyped_storage().data_ptr()
            buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
            embedding_file = f'{self.args.dst_path}/embeddings_bf16.bin'
            with open(embedding_file, 'wb') as f:
                f.write(buffer)
        elif format_bit in [8, 4]:
            # Quantized formats
            quant_bit = format_bit
            format_name = f'int{format_bit}'
            awq = getattr(self.args, 'awq', False)
            hqq = getattr(self.args, 'hqq', False)

            # Apply quantization
            q_weight, alpha = torch_quant(tensor_data.float(), quant_bit, quant_block, symmetric, awq, hqq)

            # Save quantized weights and scales together in one file
            embedding_file = f'{self.args.dst_path}/embeddings_{format_name}.bin'
            with open(embedding_file, 'wb') as f:
                weight_size = f.write(q_weight.numpy().tobytes())
                alpha_size = f.write(alpha.numpy().tobytes())
            self.llm_config['tie_embeddings'] = [0, weight_size, alpha_size, quant_bit, quant_block]
        else:
            raise ValueError(f"Unsupported embedding bit precision: {format_bit}")

        return embedding_file

    @spinner_run(f'export config to ')
    def export_config(self, mnn_config = False):
        with open(f'{self.args.dst_path}/export_args.json', 'w', encoding='utf-8') as f:
            json.dump(self.args.__dict__, f, ensure_ascii=False, indent=4)
        config_json = f'{self.args.dst_path}/llm_config.json'
        with open(config_json, 'w', encoding='utf-8') as f:
            json.dump(self.llm_config, f, ensure_ascii=False, indent=4)
        if not mnn_config:
            return config_json
        with open(f'{self.args.dst_path}/config.json', 'w', encoding='utf-8') as f:
            config = {
                "llm_model": f"{self.dst_name}.mnn",
                "llm_weight": f"{self.dst_name}.mnn.weight",
                "backend_type": "cpu",
                "thread_num": 4,
                "precision": "low",
                "memory": "low",
                # "system_prompt": "You are a helpful assistant.",
                "sampler_type": "mixed",
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.9,
                "min_p": 0.05,
                "tfs_z": 1.0,
                "typical": 0.95,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "penalty_window": 0,
                "n_gram": 8,
                "ngram_factor": 1.0
            }
            config['tokenizer_file'] = 'tokenizer.mtok'
            if self.args.embed_bit < 16:
                config['embedding_file'] = f"embeddings_int{self.args.embed_bit}.bin"
            if hasattr(self, 'talker') and self.talker is not None:
                config['system_prompt'] = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                config['talker_max_new_tokens'] = 2048
                config['talker_speaker'] = "Chelsie"
                config['dit_steps'] = 5
                config['dit_solver'] = 1
            if self.model_type == "gemma3":
                config.update({'precision': "normal"})
            if (hasattr(self, 'visual') and self.visual is not None) or (hasattr(self, 'visual') and self.audio is not None):
                config['mllm'] = {
                    'backend_type': "cpu",
                    "thread_num": 4,
                    "precision": "normal",
                    "memory": "low"
                }
            if getattr(self.args, 'visual_split', False):
                config['visual_split'] = True
                config['visual_pre_model'] = 'visual_pre.mnn'
                config['visual_blocks_model'] = 'visual_blocks.mnn'
                config['visual_post_model'] = 'visual_post.mnn'
                config['visual_blocks_backend_type'] = 'hiai'
                # K-chunk NPU split (new): populates visual_blocks_chunks list.
                # Takes priority over legacy --visual_npu_layers.
                npu_chunks = int(getattr(self.args, 'visual_npu_chunks', 0) or 0)
                if npu_chunks > 0:
                    config['visual_blocks_chunks'] = [
                        f'visual_blocks_npu_{ci}.mnn' for ci in range(npu_chunks)
                    ]
                    # Optional per-chunk backend routing for runtime mixed execution.
                    # Example: --visual_chunk_backends "npu,cpu,npu,cpu"
                    # Defaults to all-NPU if absent.
                    chunk_backends = str(getattr(self.args, 'visual_chunk_backends', '') or '').strip()
                    if chunk_backends:
                        tokens = [t.strip().lower() for t in chunk_backends.split(',') if t.strip()]
                        if len(tokens) != npu_chunks:
                            raise ValueError(
                                f"--visual_chunk_backends length ({len(tokens)}) must equal "
                                f"--visual_npu_chunks ({npu_chunks})")
                        normalized = []
                        for t in tokens:
                            if t in ('npu', 'hiai'):
                                normalized.append('npu')
                            elif t == 'cpu':
                                normalized.append('cpu')
                            else:
                                raise ValueError(
                                    f"--visual_chunk_backends only supports cpu/npu (got '{t}')")
                        config['visual_blocks_chunk_backends'] = normalized
                # Optional NPU/CPU sub-split for the blocks module (legacy 2-chunk).
                # Only added when the user opts in via --visual_npu_layers N > 0
                # AND --visual_npu_chunks is not set. Absent fields mean "use the
                # monolithic visual_blocks.mnn above".
                elif getattr(self.args, 'visual_npu_layers', 0) > 0:
                    config['visual_npu_layers'] = int(self.args.visual_npu_layers)
                    config['visual_blocks_npu_model'] = 'visual_blocks_npu.mnn'
                    config['visual_blocks_cpu_model'] = 'visual_blocks_cpu.mnn'
            if self.args.eagle_path is not None:
                config['speculative_type'] = 'eagle'
                config['hidden_states'] = True
            json.dump(config, f, ensure_ascii=False, indent=4)
        return config_json

    def imitate_quant(self):
        def quant_dequant(linear, quant_bit = self.args.quant_bit, quant_block = self.args.quant_block):
            weight = linear.weight.data
            oc, ic = weight.shape
            if quant_block == 0:
                block_size = ic
            else:
                block_size = quant_block
            block_num = ic // block_size
            weight = weight.reshape(oc, block_num, block_size)
            max_val = torch.max(weight, axis=-1, keepdims=True).values
            min_val = torch.min(weight, axis=-1, keepdims=True).values
            offset = 1 << (quant_bit - 1)
            clip_max = offset - 1
            clip_min = -offset
            scale = (max_val - min_val) / (clip_max - clip_min)
            q_weight = torch.round((weight - min_val) / scale) + clip_min
            q_weight = torch.clip(q_weight, clip_min, clip_max)
            dq_weight = (q_weight - clip_min) * scale + min_val
            dq_weight = dq_weight.reshape(oc, ic).float()
            linear.weight.data = dq_weight
            return linear
        with torch.no_grad():
            for i in range(self.config.num_hidden_layers):
                for name, child in self.model.blocks[i].self_attn.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.model.blocks[i].self_attn, name, quant_dequant(child))
                for name, child in self.model.blocks[i].mlp.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.model.blocks[i].mlp, name, quant_dequant(child))
            self.model.lm.lm = quant_dequant(self.model.lm.lm)

    def unload_param(self):
        self.unloaded_ops = {}
        self.experts = []
        self.expert_layer_ids = []
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker
        # replace linear with fakelinear to save export memory and time
        with torch.no_grad():
            for i in range(len(self.model.blocks)):
                # different kv cache shape in different layers
                # if isinstance(self.config.num_attention_heads, list):
                self.model.blocks[i].self_attn.export_fused_attn = True
                is_moe = hasattr(self.model.blocks[i].mlp, 'is_moe') and self.model.blocks[i].mlp.is_moe
                if is_moe:
                    self.model.blocks[i].mlp.export_moe = True
                for name, child in self.model.blocks[i].self_attn.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.model.blocks[i].self_attn, name, build_faker(child, f'/layers.{i}/self_attn/{name}/Linear'))
                for name, child in self.model.blocks[i].mlp.named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.model.blocks[i].mlp, name, build_faker(child, f'/layers.{i}/mlp/{name}/Linear'))
                    if name == 'shared_expert':
                        for name, child in self.model.blocks[i].mlp.shared_expert.named_children():
                            if isinstance(child, torch.nn.Linear):
                                setattr(self.model.blocks[i].mlp.shared_expert, name, build_faker(child, f'/layers.{i}/mlp/shared_expert/{name}/Linear'))
                    if is_moe and isinstance(child, torch.nn.ModuleList): # experts
                        self.experts.append(child)
                        self.expert_layer_ids.append(i)
                        for j in range(len(child)):
                            for name, cchild in child[j].named_children():
                                if isinstance(cchild, torch.nn.Linear):
                                    setattr(self.model.blocks[i].mlp.experts[j], name, build_faker(cchild, f'/expert/{i}_{j}/{name}'))
            self.model.lm.lm = build_faker(self.model.lm.lm, f'/lm/lm_head/Linear')

    @spinner_run(f'export model weight to ')
    def onnx_load_param(self, onnx_path):
        return OnnxRebuilder(onnx_path, self.unloaded_ops).rebuild()

    @spinner_run(f'slim the graph of ')
    def slim_onnx(self, onnx_model):
        import onnxslim
        model = onnxslim.slim(onnx_model)
        onnx.save(model, onnx_model)
        return onnx_model

    @spinner_run(f'export onnx model to ')
    def export_onnx(self):
        # unload linear weight to save export memory
        self.unload_param()
        model = self.model
        seq_len = 3
        new_tokens = 0
        input_ids = torch.arange(seq_len, dtype=torch.long)
        attention_mask =  model.get_attention_mask(seq_len, new_tokens)
        position_ids = model.get_position_ids(seq_len, new_tokens, input_ids)
        onnx_model = f'{self.onnx_path}/{self.dst_name}.onnx'
        # For export onnx, don't need image or audio's embedding
        input_ids = model.embedding(input_ids)
        logits_index = torch.tensor([-1], dtype=torch.int32)
        if hasattr(model, 'talker') and model.talker is not None:
            output_names = ['logits', 'hidden_states', 'talker_embeds']
        else:
            output_names = ['logits', 'hidden_states']

        # Qwen3-VL
        if self.model_type in ['qwen3_vl', 'qwen3_vl_moe']:
            # add deepstack_embeds input
            deepstack_embeds = torch.randn(3, 1, self.config.hidden_size)
            onnx_export(
                model, (input_ids, attention_mask, position_ids, logits_index, deepstack_embeds),
                onnx_model,
                input_names=[
                    'input_ids', 'attention_mask', 'position_ids', 'logits_index', 'deepstack_embeds'
                ],
                output_names=output_names,
                dynamic_axes=self.model_dynamic_axes)
            return onnx_model

        # export to onnx
        onnx_export(
            model, (input_ids, attention_mask, position_ids, logits_index),
            onnx_model,
            input_names=[
                'input_ids', 'attention_mask', 'position_ids', 'logits_index'
            ],
            output_names=output_names,
            dynamic_axes=self.model_dynamic_axes)
        return onnx_model

    def awq_quant(self):
        self.awq_quantizer = AwqQuantizer(self.model)
        self.awq_quantizer.quantize()

    def omni_quant(self):
        default_samples = 128
        total_lines = default_samples

        if self.args.calib_data:
            print(f"检测到 calib_data: {self.args.calib_data}，开始读取...")
            self.model.args.calib_data = self.args.calib_data

            if os.path.exists(self.args.calib_data):
                with open(self.args.calib_data, 'r', encoding='utf-8') as f:
                    # 统计总行数
                    total_lines = sum(1 for _ in f)
            else:
                print(f"错误：找不到文件 {self.args.calib_data}")

        calib_samples = min(total_lines, default_samples)

        print(f"OmniQuant 将使用 {calib_samples} 个样本进行优化 (Epochs={getattr(self.args, 'omni_epochs', 20)})...")

        self.omni_quantizer = OmniQuantizer(
            model=self.model,
            max_calib_samples=calib_samples,
            act_bit=self.args.act_bit,
            act_sym=self.args.act_sym,
            generate_for_npu=self.args.generate_for_npu,

            epochs=getattr(self.args, 'omni_epochs', 20),
            lr=getattr(self.args, 'omni_lr', 5e-3),
            wd=getattr(self.args, 'omni_wd', 1e-4)
        )
        self.omni_quantizer.quantize(self.args.generate_for_npu)

    def smooth_quant(self):
        total_lines = 128
        if self.args.calib_data:
            print(f"检测到 calib_data: {self.args.calib_data}，开始读取...")
            self.model.args.calib_data = self.args.calib_data

            if os.path.exists(self.args.calib_data):
                with open(self.args.calib_data, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)
            else:
                print(f"错误：找不到文件 {self.args.calib_data}")

        calib_samples = min(total_lines, 128)
        self.smooth_quantizer = SmoothQuantizer(model = self.model, max_calib_samples = calib_samples, act_bit=self.args.act_bit, act_sym=self.args.act_sym, generate_for_npu=self.args.generate_for_npu)
        self.smooth_quantizer.quantize()

    def _build_visual_split_wrappers(self):
        """Wrap self.visual into 3 torch.nn.Modules (pre / blocks / post) that share
        its weights. Each wrapper handles Qwen2Vision / Qwen3_5Vision / Qwen3Vision
        variants by probing attributes on the underlying instance.

        Returns (pre, blocks, post, meta) where meta is a dict describing I/O
        names, dummy shapes and whether deepstack is enabled. The wrappers run
        exactly the same sub-computation as self.visual.forward, so per-block
        GPTQ indices and weight naming are preserved across the 3 ONNX files.
        """
        visual = self.visual
        has_pos_embed = hasattr(visual, 'pos_embed') and visual.pos_embed is not None
        has_deepstack = (hasattr(visual, 'deepstack_visual_indexes')
                         and visual.deepstack_visual_indexes is not None
                         and len(visual.deepstack_visual_indexes) > 0
                         and hasattr(visual, 'deepstack_merger_list'))
        unsqueeze_output = hasattr(visual, 'num_grid_per_side')  # Qwen3_5Vision / Qwen3Vision

        class _VisualPre(torch.nn.Module):
            def __init__(self, vis):
                super().__init__()
                self.rotary = vis.rotary
                self.patch_embed = vis.patch_embed
                if has_pos_embed:
                    self.pos_embed = vis.pos_embed

            def forward(self, flatten_patches, position_ids, idx_tensor=None, weight_tensor=None):
                rotary_pos_emb = self.rotary(position_ids)
                hidden_states = self.patch_embed(flatten_patches)
                if has_pos_embed:
                    pos_embeds = self.pos_embed(idx_tensor) * weight_tensor.unsqueeze(2)
                    pos_embeds = torch.sum(pos_embeds, 0, False)
                    hidden_states = hidden_states + pos_embeds
                if rotary_pos_emb.dtype != hidden_states.dtype:
                    rotary_pos_emb = rotary_pos_emb.to(hidden_states.dtype)
                return hidden_states, rotary_pos_emb

        class _VisualBlocks(torch.nn.Module):
            def __init__(self, vis):
                super().__init__()
                self.blocks = torch.nn.ModuleList(vis.blocks)
                if has_deepstack:
                    self.deepstack_visual_indexes = list(vis.deepstack_visual_indexes)

            def forward(self, hidden_states, rotary_pos_emb, attention_mask):
                deepstack_hidden = []
                for layer_num, blk in enumerate(self.blocks):
                    hidden_states = blk(hidden_states, rotary_pos_emb=rotary_pos_emb,
                                        attention_mask=attention_mask)
                    if has_deepstack and layer_num in self.deepstack_visual_indexes:
                        deepstack_hidden.append(hidden_states)
                if has_deepstack:
                    return (hidden_states, *deepstack_hidden)
                return hidden_states

        class _VisualPost(torch.nn.Module):
            def __init__(self, vis):
                super().__init__()
                self.merger = vis.merger
                if has_deepstack:
                    self.deepstack_merger_list = torch.nn.ModuleList(vis.deepstack_merger_list)

            def forward(self, hidden_states, *deepstack_hidden):
                if has_deepstack:
                    feats = [m(h) for m, h in zip(self.deepstack_merger_list, deepstack_hidden)]
                    deepstack_feature = torch.stack(feats)
                image_embeds = self.merger(hidden_states)
                if unsqueeze_output:
                    image_embeds = image_embeds.unsqueeze(1)
                if has_deepstack:
                    return image_embeds, deepstack_feature
                return image_embeds

        # Choose dummy shapes. Prefer the same ones used by visual.export().
        if has_pos_embed:
            seq_len = 256
            patches_dim = 1536
            rotary_feat = seq_len  # rotary_pos_emb leading dim equals seq_len
        else:
            seq_len = 256
            patches_dim = 1176  # Qwen2Vision default patch dim; rarely used in split mode
            rotary_feat = seq_len

        meta = {
            'has_pos_embed': has_pos_embed,
            'has_deepstack': has_deepstack,
            'unsqueeze_output': unsqueeze_output,
            'seq_len': seq_len,
            'patches_dim': patches_dim,
            'deepstack_indexes': list(visual.deepstack_visual_indexes) if has_deepstack else [],
        }
        return _VisualPre(visual), _VisualBlocks(visual), _VisualPost(visual), meta

    def _build_visual_blocks_chunk(self, start_layer, end_layer, has_deepstack, deepstack_indexes):
        """Build a torch.nn.Module wrapping visual.blocks[start_layer:end_layer].
        Same I/O signature as _VisualBlocks (hidden_states, rotary_pos_emb, attention_mask
        -> hidden_states [, deepstack_hidden_0, ..., deepstack_hidden_M-1]), but only
        the deepstack layers whose GLOBAL index falls in [start_layer, end_layer) are
        captured. This preserves original ordering when the caller concatenates outputs
        from the NPU chunk and the CPU chunk.
        Non-invasive: callers that don't use this stay on _VisualBlocks.
        """
        visual = self.visual
        # Absolute layer index → True if that layer emits a deepstack_hidden.
        # Filter to only indexes inside this chunk.
        local_indexes = []
        if has_deepstack:
            local_indexes = [i - start_layer for i in deepstack_indexes
                             if start_layer <= i < end_layer]

        class _VisualBlocksChunk(torch.nn.Module):
            def __init__(self, vis, s, e):
                super().__init__()
                self.blocks = torch.nn.ModuleList(vis.blocks[s:e])
                self._local_ds = list(local_indexes)
                self._has_ds = bool(local_indexes)

            def forward(self, hidden_states, rotary_pos_emb, attention_mask):
                ds = []
                for layer_num, blk in enumerate(self.blocks):
                    hidden_states = blk(hidden_states, rotary_pos_emb=rotary_pos_emb,
                                        attention_mask=attention_mask)
                    if self._has_ds and layer_num in self._local_ds:
                        ds.append(hidden_states)
                if self._has_ds:
                    return (hidden_states, *ds)
                return hidden_states

        return _VisualBlocksChunk(visual, start_layer, end_layer), local_indexes

    @torch.no_grad()
    def export_vision_split(self):
        """Export visual as 3 separate .mnn files: pre / blocks / post.

        Intended use: run 'blocks' on NPU, pre/post on CPU. Block I/O is 3D
        [B, S, D] so no NC4HW4 layout conversion is needed at the boundary.
        """
        pre, blocks, post, meta = self._build_visual_split_wrappers()
        pre.eval(); blocks.eval(); post.eval()

        def _reset_kv_cache(m):
            # Attention modules in utils/transformers.py cache K/V in
            # self.past_key_value across calls, which makes a dry-run double K/V
            # seq length on the subsequent trace. Force-clear it before every
            # forward pass we plan to export.
            for sub in m.modules():
                if hasattr(sub, 'past_key_value'):
                    sub.past_key_value = None

        S = meta['seq_len']
        D_in = meta['patches_dim']
        # Shape of hidden_states after patch_embed; we can't know D_out without a
        # dry run, so do one now to capture it (shapes only; weights are real).
        patches = torch.randn([S, D_in])
        position_ids = torch.zeros([2, S], dtype=torch.int32)
        _reset_kv_cache(pre)
        if meta['has_pos_embed']:
            idx_tensor = torch.zeros([4, S], dtype=torch.int32)
            weight_tensor = torch.randn([4, S])
            hidden_states, rotary_pos_emb = pre(patches, position_ids, idx_tensor, weight_tensor)
        else:
            hidden_states, rotary_pos_emb = pre(patches, position_ids)
        # patch_embed can change the seq dim (temporal fold/unfold). Build the
        # attention_mask from the actual hidden_states length.
        S_out = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[1]
        attention_mask = torch.zeros([1, S_out, S_out], dtype=hidden_states.dtype)
        _reset_kv_cache(blocks)
        blocks_out = blocks(hidden_states, rotary_pos_emb, attention_mask)
        if isinstance(blocks_out, tuple):
            hidden_final = blocks_out[0]
        else:
            hidden_final = blocks_out

        # --- Export pre to ONNX ---
        pre_onnx = os.path.join(self.onnx_path, 'visual_pre.onnx')
        _reset_kv_cache(pre)
        if meta['has_pos_embed']:
            onnx_export(pre,
                        (patches, position_ids, idx_tensor, weight_tensor),
                        pre_onnx,
                        input_names=['patches', 'position_ids', 'idx_tensor', 'weight_tensor'],
                        output_names=['hidden_states', 'rotary_pos_emb'],
                        dynamic_axes={
                            'patches': {0: 'size'},
                            'position_ids': {1: 'size'},
                            'idx_tensor': {1: 'size'},
                            'weight_tensor': {1: 'size'},
                            'hidden_states': {0: 'size'},
                            'rotary_pos_emb': {0: 'size'},
                        })
        else:
            onnx_export(pre,
                        (patches, position_ids),
                        pre_onnx,
                        input_names=['patches', 'position_ids'],
                        output_names=['hidden_states', 'rotary_pos_emb'],
                        dynamic_axes={
                            'patches': {0: 'size'},
                            'position_ids': {1: 'size'},
                            'hidden_states': {0: 'size'},
                            'rotary_pos_emb': {0: 'size'},
                        })

        # --- Export blocks to ONNX ---
        # When --visual_npu_layers > 0 or --visual_npu_chunks > 0 the runtime
        # only uses the chunk files (visual_blocks_npu*.mnn [+ visual_blocks_cpu.mnn])
        # — the monolithic visual_blocks.mnn would just be dead weight on disk AND
        # building it through self._convert_visual_piece(...) is as memory-heavy
        # as the thing we're trying to avoid. Skip it in that case.
        _skip_monolithic_blocks = (
            int(getattr(self.args, 'visual_npu_layers', 0) or 0) > 0
            or int(getattr(self.args, 'visual_npu_chunks', 0) or 0) > 0
        )
        blocks_onnx = None
        if not _skip_monolithic_blocks:
            blocks_onnx = os.path.join(self.onnx_path, 'visual_blocks.onnx')
            _reset_kv_cache(blocks)
            blocks_outputs = ['hidden_states']
            blocks_dyn = {
                'hidden_states_in': {0: 'size'},
                'rotary_pos_emb': {0: 'size'},
                'attention_mask': {1: 'size', 2: 'size'},
                'hidden_states': {0: 'size'},
            }
            if meta['has_deepstack']:
                for i in range(len(meta['deepstack_indexes'])):
                    name = f'deepstack_hidden_{i}'
                    blocks_outputs.append(name)
                    blocks_dyn[name] = {0: 'size'}
            onnx_export(blocks,
                        (hidden_states, rotary_pos_emb, attention_mask),
                        blocks_onnx,
                        input_names=['hidden_states_in', 'rotary_pos_emb', 'attention_mask'],
                        output_names=blocks_outputs,
                        dynamic_axes=blocks_dyn)

        # --- Export post to ONNX ---
        post_onnx = os.path.join(self.onnx_path, 'visual_post.onnx')
        _reset_kv_cache(post)
        if meta['has_deepstack']:
            deepstack_dummy = tuple(hidden_final for _ in meta['deepstack_indexes'])
            in_names = ['hidden_states'] + [f'deepstack_hidden_{i}' for i in range(len(meta['deepstack_indexes']))]
            out_names = ['image_embeds', 'deepstack_feature']
            dyn = {n: {0: 'size'} for n in in_names}
            onnx_export(post,
                        (hidden_final, *deepstack_dummy),
                        post_onnx,
                        input_names=in_names,
                        output_names=out_names,
                        dynamic_axes=dyn)
        else:
            onnx_export(post,
                        (hidden_final,),
                        post_onnx,
                        input_names=['hidden_states'],
                        output_names=['image_embeds'],
                        dynamic_axes={'hidden_states': {0: 'size'}})

        # --- Convert each piece to MNN using the selected quant path ---
        if self.mnn_converter:
            self._convert_visual_piece(pre_onnx)
            if blocks_onnx is not None:
                self._convert_visual_piece(blocks_onnx)
            self._convert_visual_piece(post_onnx)

        # --- Optional NPU chunk export of blocks ---
        # Two activation modes (mutually exclusive; --visual_npu_chunks takes priority):
        #
        #  (1) --visual_npu_chunks K > 0  (new):
        #      Split the blocks into K roughly-equal NPU chunks. Emits
        #        visual_blocks_npu_0.mnn ... visual_blocks_npu_{K-1}.mnn
        #      all targeting the same NPU runtime. Each chunk is O(1/K) the
        #      weights of the monolithic module, so HiAI IR-build peak memory
        #      scales down by K. Runtime chains them in order.
        #
        #  (2) --visual_npu_layers N > 0  (legacy 2-chunk):
        #      First N layers → visual_blocks_npu.mnn (NPU),
        #      Remaining layers → visual_blocks_cpu.mnn (CPU).
        #      File names preserved for back-compat with existing configs.
        #
        # Neither set: skip entirely (the monolithic visual_blocks.mnn above is
        # the only blocks artifact).
        npu_chunks = int(getattr(self.args, 'visual_npu_chunks', 0) or 0)
        npu_layers = int(getattr(self.args, 'visual_npu_layers', 0) or 0)
        if npu_chunks > 0 or npu_layers > 0:
            total = len(self.visual.blocks)

            # Resolve chunk boundaries and per-chunk file names.
            # Each entry is (start, end, onnx_basename, is_last_cpu_only).
            chunk_specs = []
            if npu_chunks > 0:
                if npu_chunks < 2:
                    raise ValueError(f"--visual_npu_chunks must be >= 2 (got {npu_chunks}); "
                                     f"use --visual_npu_chunks=0 + monolithic for K=1")
                if npu_chunks > total:
                    raise ValueError(f"--visual_npu_chunks={npu_chunks} must be <= total blocks ({total})")
                # Equal split with the remainder absorbed into the last chunk,
                # so chunk sizes differ by at most 1.
                base = total // npu_chunks
                rem = total % npu_chunks
                cursor = 0
                for ci in range(npu_chunks):
                    size = base + (1 if ci < rem else 0)
                    s, e = cursor, cursor + size
                    chunk_specs.append((s, e, f'visual_blocks_npu_{ci}.onnx', False))
                    cursor = e
                assert cursor == total
            else:
                # Legacy 2-chunk: NPU first-N + CPU rest. Preserve old names.
                if npu_layers >= total:
                    raise ValueError(f"--visual_npu_layers={npu_layers} must be < total blocks ({total})")
                chunk_specs.append((0, npu_layers, 'visual_blocks_npu.onnx', False))
                chunk_specs.append((npu_layers, total, 'visual_blocks_cpu.onnx', True))

            # Export each chunk. Between chunks, dry-run to get the next chunk's
            # hidden_states input. rotary_pos_emb and attention_mask are reused.
            last_hidden = hidden_states
            global_ds_cursor = 0
            chunk_onnx_paths = []
            for (s, e, fname, _is_cpu_tail) in chunk_specs:
                chunk_module, local_ds_idx = self._build_visual_blocks_chunk(
                    s, e, meta['has_deepstack'], meta['deepstack_indexes'])
                chunk_module.eval()

                chunk_onnx = os.path.join(self.onnx_path, fname)
                _reset_kv_cache(chunk_module)
                out_names = ['hidden_states']
                dyn = {
                    'hidden_states_in': {0: 'size'},
                    'rotary_pos_emb': {0: 'size'},
                    'attention_mask': {1: 'size', 2: 'size'},
                    'hidden_states': {0: 'size'},
                }
                # Deepstack outputs use GLOBAL indices, matching what the post
                # module expects (deepstack_hidden_0 .. _{M-1}). Each chunk emits
                # only the subset whose global layer index falls in [s, e).
                for k in range(len(local_ds_idx)):
                    name = f'deepstack_hidden_{global_ds_cursor + k}'
                    out_names.append(name)
                    dyn[name] = {0: 'size'}
                onnx_export(chunk_module,
                            (last_hidden, rotary_pos_emb, attention_mask),
                            chunk_onnx,
                            input_names=['hidden_states_in', 'rotary_pos_emb', 'attention_mask'],
                            output_names=out_names,
                            dynamic_axes=dyn)
                chunk_onnx_paths.append(chunk_onnx)

                # Dry-run to propagate hidden_states into the next chunk.
                # ViT blocks are shape-preserving, so this only fixes input
                # tensor values; shapes don't drift.
                _reset_kv_cache(chunk_module)
                chunk_out = chunk_module(last_hidden, rotary_pos_emb, attention_mask)
                last_hidden = chunk_out[0] if isinstance(chunk_out, tuple) else chunk_out
                global_ds_cursor += len(local_ds_idx)

            if self.mnn_converter:
                for p in chunk_onnx_paths:
                    self._convert_visual_piece(p)

    def _convert_visual_piece(self, onnx_path):
        """Route a single visual-split onnx file through whichever MNN conversion
        path the user asked for (same flags as the non-split export_vision)."""
        fuse_transformer = self.visual.transformer_fuse
        native_group_conv = self.visual.group_conv_native
        quant_bit_visual = self.visual.quant_bit
        quant_block_visual = self.visual.quant_block
        if self.args.transformer_fuse:
            fuse_transformer = True
        if self.args.group_conv_native:
            native_group_conv = True
        if self.args.visual_quant_bit is not None:
            quant_bit_visual = self.args.visual_quant_bit
        if self.args.visual_quant_block is not None:
            quant_block_visual = self.args.visual_quant_block
        if self.args.visual_keep_matmul:
            self.mnn_converter.export_visual_fp16_matmul(
                onnx_path,
                transformer_fuse=fuse_transformer,
                group_conv_native=native_group_conv,
                weight_sym=self.args.visual_sym,
                slim_json=not self.args.visual_json_full,
                emit_json=not self.args.visual_no_json,
            )
        elif self.args.visual_gptq_path is not None:
            self.mnn_converter.export_visual_with_gptq(
                onnx_path,
                self.args.visual_gptq_path,
                quant_block=quant_block_visual if quant_block_visual else 128,
                transformer_fuse=fuse_transformer,
                group_conv_native=native_group_conv,
                weight_sym=self.args.visual_sym,
            )
        else:
            self.mnn_converter.export(
                onnx_path, quant_bit_visual, quant_block_visual,
                transformer_fuse=fuse_transformer,
                group_conv_native=native_group_conv,
                weight_sym=self.args.visual_sym,
            )

    def export_vision(self):
        if self.visual is None:
            return
        if getattr(self.args, 'visual_split', False):
            return self.export_vision_split()
        vision_onnx = self.visual.export(self.onnx_path)
        if self.mnn_converter:
            fuse_transformer = self.visual.transformer_fuse
            native_group_conv = self.visual.group_conv_native
            quant_bit_visual = self.visual.quant_bit
            quant_block_visual = self.visual.quant_block
            if self.args.transformer_fuse:
                fuse_transformer = True
            if self.args.group_conv_native:
                native_group_conv = True
            if self.args.visual_quant_bit is not None:
                quant_bit_visual = self.args.visual_quant_bit
            if self.args.visual_quant_block is not None:
                quant_block_visual = self.args.visual_quant_block
            if self.args.visual_keep_matmul:
                # fp16 + MatMul preserved (no MatMul->Conv fold). No GPTQ applied.
                self.mnn_converter.export_visual_fp16_matmul(
                    vision_onnx,
                    transformer_fuse=fuse_transformer,
                    group_conv_native=native_group_conv,
                    weight_sym=self.args.visual_sym,
                    slim_json=not self.args.visual_json_full,
                    emit_json=not self.args.visual_no_json
                )
            elif self.args.visual_gptq_path is not None:
                # GPTQ path: onnx2mnn(fp16) -> mnn2json -> apply visual gptq(blocks->int8) -> json2mnn
                # merger/deepstack/patch_embed keep fp16, only blocks are replaced with GPTQ int8
                self.mnn_converter.export_visual_with_gptq(
                    vision_onnx,
                    self.args.visual_gptq_path,
                    quant_block=quant_block_visual if quant_block_visual else 128,
                    transformer_fuse=fuse_transformer,
                    group_conv_native=native_group_conv,
                    weight_sym=self.args.visual_sym
                )
            else:
                self.mnn_converter.export(vision_onnx, quant_bit_visual,
                                          quant_block_visual,
                                          transformer_fuse=fuse_transformer,
                                          group_conv_native=native_group_conv,
                                          weight_sym=self.args.visual_sym)

    def export_audio(self):
        if self.audio is None:
            return
        audio_onnx = self.audio.export(self.onnx_path)
        if self.mnn_converter: self.mnn_converter.export(audio_onnx, self.audio.quant_bit)

    def export_talker(self):
        if self.talker is None:
            return
        talker_onnx = self.talker.export(self.onnx_path)
        predit_onnx, dit_onnx, bigvgan_onnx = self.talker.token2wav.export(self.onnx_path)
        if self.mnn_converter:
            self.mnn_converter.export(talker_onnx, self.talker.quant_bit)
            self.mnn_converter.export(predit_onnx, self.talker.token2wav.quant_bit)
            self.mnn_converter.export(dit_onnx, self.talker.token2wav.quant_bit)
            self.mnn_converter.export(bigvgan_onnx, self.talker.token2wav.quant_bit)

    def export_language(self):
        # export_embedding
        if self.mnn_converter and self.args.tie_word_embeddings:
            print("tie_word_embeddings")
            pass # mnn tie_word_embeddings need't export embedding
        else:
            self.export_embed()
        # export transformer
        onnx_model = self.export_onnx()

        if self.args.onnx_slim:
            self.slim_onnx(onnx_model)
        if self.mnn_converter:
            tie_embeddings_info = MNNConverter(self, self.unloaded_ops).export(onnx_model)
            if tie_embeddings_info is not None:
                self.llm_config['tie_embeddings'] = tie_embeddings_info
        else:
            self.onnx_load_param(onnx_model)

    def export(self, export_type):
        if not self.args.skip_weight:
            if self.args.omni:
                self.omni_quant()
            if self.args.awq:
                self.awq_quant()
            if self.args.smooth:
                self.smooth_quant()
        export_mnn = export_type == 'mnn'
        self.mnn_converter = MNNConverter(self) if export_mnn else None
        self.export_talker()
        self.export_vision()
        self.export_audio()
        self.export_eagle()
        self.export_language()
        self.export_mtp()
        self.export_tokenizer()
        self.export_config(export_mnn)
        if export_mnn and self.args.cleanup_onnx:
            # delete onnx file
            try:
                for file in glob.glob(f'{self.onnx_path}/*'):
                    os.remove(file)
                os.rmdir(self.onnx_path)
            except Exception as e:
                print(f"remove onnx error: {e}")

    @spinner_run(f'export tokenizer to ')
    def export_tokenizer(self):
        return self.tokenizer.export(self.args.dst_path)

class EmbeddingExporter(LlmExporter):
    def __init__(self, args):
        super().__init__(args)

    def response(self, query):
        self.model.eval()
        prompt = self.build_prompt(query)
        input_ids = self.tokenizer(prompt)['input_ids']
        seq_len = len(input_ids)
        input_ids = torch.tensor(input_ids)
        position_ids = self.model.get_position_ids(seq_len)
        attention_mask = self.model.get_attention_mask(seq_len)
        inputs_embeds = self.model.word_embed(input_ids)
        res = self.model.forward(inputs_embeds, attention_mask, position_ids)
        print(res, res.shape)
        return res

    def build_prompt(self, content):
        if self.config.model_type == 'bert':
            return f'[CLS]{content}[SEP]'
        if self.config.model_type == 'new':
            return f'<s> {content}</s>'
        if self.config.model_type == 'qwen3':
            return f'{content}<|endoftext|>'

    @spinner_run(f'load pretrained model ', True)
    def load_model(self, model_path):
        self.model = EmbeddingModel.from_pretrained(model_path, args=self.args)
        self.config = self.model.config
        self.model_type = self.config.model_type
        self.tokenizer = LlmTokenizer(model_path, self.model_type)
        self.llm_config = {
            'model_type': self.config.model_type,
            'hidden_size' : self.config.hidden_size,
            'attention_mask': 'int',
            "jinja": {
                "chat_template": self.build_prompt("{{ messages | map(attribute='content') | join('') }}")
            },
            'is_visual': False
        }
        return model_path

    def export_reranker(self):
        seq_len = 4
        input_ids = torch.arange(12, dtype=torch.long)
        position_ids = self.model.get_position_ids(seq_len)
        attention_mask = self.model.get_attention_mask(seq_len)
        inputs_embeds = self.model.word_embed(input_ids)
        inputs_embeds = inputs_embeds.reshape(3, 4, self.config.hidden_size)
        attention_mask = torch.zeros(3, 1, 1, 4, dtype=torch.float)
        onnx_model = f'{self.onnx_path}/{self.dst_name}.onnx'
        onnx_export(
            self.model, (inputs_embeds, attention_mask, position_ids),
            onnx_model,
            input_names=[
                'input_ids',
                'attention_mask',
                'position_ids'
            ],
            output_names=['sentence_embeddings'],
            dynamic_axes={
                "input_ids" : { 0: "batch", 1: "seq_len" },
                "position_ids" : { 1: "seq_len" },
                "attention_mask" : { 0: "batch", 3: "seq_len" }
            })
        return onnx_model

    @spinner_run(f'export onnx model to ')
    def export_onnx(self):
        if self.model_type == 'qwen3':
            self.unload_param()
        else:
            self.unloaded_ops = None
        if self.model.is_reranker:
            return self.export_reranker()
        seq_len = 3
        input_ids = torch.arange(seq_len, dtype=torch.long)
        position_ids = self.model.get_position_ids(seq_len)
        attention_mask = self.model.get_attention_mask(seq_len)
        inputs_embeds = self.model.word_embed(input_ids)
        onnx_model = f'{self.onnx_path}/{self.dst_name}.onnx'
        onnx_export(
            self.model, (inputs_embeds, attention_mask, position_ids),
            onnx_model,
            input_names=[
                'input_ids',
                'attention_mask',
                'position_ids'
            ],
            output_names=['sentence_embeddings'],
            dynamic_axes={
                "input_ids" : { 1: "seq_len" },
                "position_ids" : { 1: "seq_len" },
                "attention_mask" : { 2: "seq_len", 3: "seq_len" }
            })
        return onnx_model

    def export(self, export_type):
        export_mnn = 'mnn' in export_type
        self.export_tokenizer()
        self.export_embed()
        self.export_config(export_mnn)
        onnx_model = self.export_onnx()
        if self.args.onnx_slim:
            self.slim_onnx(onnx_model)
        if export_mnn:
            transformer_fuse = not self.model.is_reranker
            tie_embeddings_info = MNNConverter(self, self.unloaded_ops).export(onnx_model, transformer_fuse=transformer_fuse)
            if tie_embeddings_info is not None:
                self.llm_config['tie_embeddings'] = tie_embeddings_info
            if self.args.cleanup_onnx:
                # delete onnx file
                try:
                    for file in glob.glob(f'{self.onnx_path}/*'):
                        os.remove(file)
                    os.rmdir(self.onnx_path)
                except Exception as e:
                    print(f"remove onnx error: {e}")

def build_args(parser):
    parser.add_argument('--path', type=str, required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--type', type=str, default=None,
                        help='type(`str`, *optional*):'
                        '\n\tThe pretrain llm model type.'
                        )
    parser.add_argument('--tokenizer_path', type=str, default=None, help='tokenizer path, default is `None` mean using `--path` value.')
    parser.add_argument('--eagle_path', type=str, default=None, help='eagle model path, default is `None`')
    parser.add_argument('--lora_path', type=str, default=None, help='lora path, default is `None` mean not apply lora.')
    parser.add_argument('--gptq_path', type=str, default=None, help='gptq path, default is `None` mean not apply gptq.')
    parser.add_argument('--visual_gptq_path', type=str, default=None, help='gptq path for visual model, default is `None` mean not apply visual gptq.')
    parser.add_argument('--visual_keep_matmul', action='store_true', help='Export visual model as fp16 with MatMul preserved (no MatMul->Conv fold). No GPTQ applied. Overrides --visual_gptq_path.')
    parser.add_argument('--visual_json_full', action='store_true', help='Keep full (un-slimmed) visual.mnn.json with inline weight arrays. Only effective with --visual_keep_matmul. Default: slim (strip weight arrays).')
    parser.add_argument('--visual_no_json', action='store_true', help='Skip visual.mnn.json generation entirely (mnn2json step is memory-heavy on large fp16 models). Only effective with --visual_keep_matmul.')
    parser.add_argument('--visual_split', action='store_true', help='Export visual model as 3 separate .mnn files (visual_pre.mnn / visual_blocks.mnn / visual_post.mnn) so blocks can run on NPU while pre/post run on CPU. Uses the same quant flags as the non-split path.')
    parser.add_argument('--visual_npu_layers', type=int, default=0, help='[TEMPORARY NPU TEST] With --visual_split, put the first N transformer blocks into visual_blocks_npu.mnn (for NPU) and the remaining into visual_blocks_cpu.mnn (for CPU). Default 0 = disabled, existing --visual_split behaviour preserved. Non-invasive: pre/post and DeepStack merging are unchanged; merely shards the blocks wrapper so the NPU build phase is cheap to test.')
    parser.add_argument('--visual_npu_chunks', type=int, default=0, help='With --visual_split, split the visual transformer blocks into K roughly equal NPU chunks (visual_blocks_npu_0.mnn ... visual_blocks_npu_{K-1}.mnn) so each HiAI IR-build processes only 1/K of the weights. Recommended when the monolithic build OOMs. Overrides --visual_npu_layers when > 0. Default 0 = disabled. Example: --visual_npu_chunks 4 on a 24-block ViT produces 4 chunks of 6 blocks each.')
    parser.add_argument('--visual_chunk_backends', type=str, default='', help='Optional per-chunk backend routing for --visual_npu_chunks. Comma-separated list with length == K. Allowed tokens: cpu,npu (or hiai). Example: --visual_npu_chunks 6 --visual_chunk_backends "npu,npu,cpu,cpu,npu,cpu".')
    parser.add_argument('--dst_path', type=str, default='./model', help='export onnx/mnn model to path, default is `./model`.')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to print verbose.')
    parser.add_argument('--test', type=str, help='test model inference with query `TEST`.')
    parser.add_argument('--export', type=str, default=None, help='export model to an onnx/mnn model.')
    parser.add_argument('--onnx_slim', action='store_true', help='Whether or not to use onnx-slim.')
    parser.add_argument('--cleanup_onnx', action='store_true', help='Delete intermediate onnx files after export.')
    parser.add_argument('--quant_bit', type=int, default=4, help='mnn quant bit, 4 or 8, default is 4.')
    parser.add_argument('--quant_block', type=int, default=64, help='mnn quant block, 0 mean channel-wise, default is 64.')
    parser.add_argument('--visual_quant_bit', type=int, default=None, help='mnn visual quant bit, 4 or 8, default is setting in utils/vision.py by different vit model.')
    parser.add_argument('--visual_quant_block', type=int, default=None, help='mnn quant block, default is setting in utils/vision.py by different vit model.')
    parser.add_argument('--lm_quant_bit', type=int, default=None, help='mnn lm_head quant bit, 4 or 8, default is `quant_bit`.')
    parser.add_argument('--lm_quant_block', type=int, default=None, help='mnn lm_head quant block, 0 mean channle-wise, default is `quant_block`.')
    parser.add_argument('--mnnconvert', type=str, default='../../../build/MNNConvert', help='local mnnconvert path, if invalid, using pymnn.')
    parser.add_argument('--ppl', action='store_true', help='Whether or not to get all logits of input tokens.')
    parser.add_argument('--awq', action='store_true', help='Whether or not to use awq quant.')
    parser.add_argument('--hqq', action='store_true', help='Whether or not to use hqq quant.')
    parser.add_argument('--omni', action='store_true', help='Whether or not to use omni quant.')
    parser.add_argument('--transformer_fuse', action='store_true', help='Whether or not to fuse vision transformer op.')
    parser.add_argument('--group_conv_native', action='store_true', help='Whether or not to keep native group_conv.')
    parser.add_argument('--smooth', action='store_true', help='Whether or not to use smooth quant.')
    parser.add_argument('--sym', action='store_true', help='Whether or not to using symmetric quant (without zeropoint), default is False.')
    parser.add_argument('--visual_sym', action='store_true', help='Whether or not to using symmetric quant (without zeropoint) for visual model, default is False.')
    parser.add_argument('--seperate_embed', action='store_true', help='For lm and embed shared model, whether or not to sepearte embed to avoid quant, default is False, if True, embed weight will be seperate to embedding bf16.bin.')
    parser.add_argument('--lora_split', action='store_true', help='Whether or not export lora split, default is False.')
    parser.add_argument('--calib_data', type=str, default=None, help='calibration data path, default is `None` mean not use calib data.')
    parser.add_argument('--act_bit', type=int, default=16, help='smooth quant act bit, 8 or 16, default is 16.')
    parser.add_argument('--embed_bit', type=int, default=16, choices=[16, 8, 4], help='embedding export bit precision, choices are 16 (bf16), 8 (int8), 4 (int4), default is 16.')
    parser.add_argument('--act_sym', action='store_true', help='smooth quant act us sym or not, default asym.')
    parser.add_argument('--quant_config', type=str, default=None, help='path to the JSON file for op-wise quantization configuration.')
    parser.add_argument('--generate_for_npu', action='store_true', help='Whether or not to generate model for NPU deployment, default is False.')
    parser.add_argument('--skip_weight', action='store_true', help='Whether or not to skip loading model weights, useful for testing export flow.')
    # omni quant
    parser.add_argument('--omni_epochs', type=int, default=20, help='OmniQuant 优化的轮数')
    parser.add_argument('--omni_lr', type=float, default=5e-3, help='OmniQuant 的学习率')
    parser.add_argument('--omni_wd', type=float, default=1e-4, help='OmniQuant 的权重衰减')

def export(path, **kwargs):
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args(['--path', path])
    for k, v in kwargs.items():
        setattr(args, k, v)
    if 'bge' in path:
        llm_exporter = EmbeddingExporter(args)
    else:
        llm_exporter = LlmExporter(args)
    # export
    llm_exporter.export(args.export)

def main():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    build_args(parser)
    args = parser.parse_args()

    model_path = args.path

    embedding_models = ['bge', 'gte', 'Qwen3-Embedding']
    if any(model in model_path for model in embedding_models):
        llm_exporter = EmbeddingExporter(args)
    else:
        llm_exporter = LlmExporter(args)

    # some actions
    if args.test is not None:
        llm_exporter.response(args.test)

    if args.export is not None:
        print('export model to', args.export)
        llm_exporter.export(args.export)

if __name__ == '__main__':
    main()
