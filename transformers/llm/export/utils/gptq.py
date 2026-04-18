import glob
import json
import numpy as np
import torch
from safetensors import safe_open

class GPTQWeight:
    def __init__(self, name):
        self.name = name

    def __repr__(self) -> str:
        if hasattr(self, 'qweight'):
            return f'{self.name}, {self.qweight.shape}, {self.scales.shape}'
        return 'None'

    def add(self, name, tensor):
        setattr(self, name, tensor)

    def weight(self, idx):
        shape = self.qweight.shape
        if len(shape) == 2:
            ic, oc = shape
            self.qweight = self.qweight.reshape(ic//16, 16, oc)
        return self.qweight[idx]

    def scale(self, idx):
        return self.scales[idx]

class MNNWeight:
    def __init__(self, name, external, weight_elements):
        self.name = name
        self.external = external
        self.quant_bits = 4
        if round(weight_elements / external[1]) == 2:
            self.quant_bits = 4
            self.a_min = -8
        else:
            self.quant_bits = 8
            self.a_min = -128
        self.parse_name()

    def __repr__(self) -> str:
        return f'{self.layer_id}.{self.op_id}.{self.block_id}, {self.external}'

    def parse_name(self):
        parts = self.name.split('/')
        if len(parts) > 4:
            self.layer_id = parts[1].split('.')[1]
            self.op_id = parts[2] + '.' + parts[3]
            self.block_id = parts[-1].split('__')[-1]
        else:
            self.layer_id = -1
            self.op_id = parts[2]
            self.block_id = parts[-1].split('__')[-1]

    def key(self):
        if self.layer_id == -1: return self.op_id
        return f'{self.layer_id}.{self.op_id}'
    def offset(self): return self.external[0]
    def weight_size(self): return self.external[1]
    def scale_size(self): return self.external[2]

class GPTQ:
    def __init__(self, gptq_path):
        self.load(gptq_path)

    def load(self, path):
        for tensor in glob.glob(f'{path}/*.safetensors'):
            self.load_safetensor(tensor)

    def prefix(self, name):
        """从完整键名中提取 MNN 兼容的模块前缀（格式: {layer_idx}.{module_type}.{proj_name}）"""
        splits = name.split('.')
        
        # 处理 lm_head (MNN 可能不需要，但保持兼容)
        if 'lm_head' in name and len(splits) >= 2 and splits[-1] in ['qweight', 'scales']:
            return 'lm_head', splits[-1]
        
        # 关键修复：精确解析 layers.X.{module}.{proj}.qweight/scales
        try:
            # 查找 'layers' 的位置（兼容 model.language_model.layers 或 model.layers）
            layers_idx = -1
            for i, part in enumerate(splits):
                if part == 'layers':
                    layers_idx = i
                    break
            if layers_idx == -1 or layers_idx + 3 >= len(splits):
                return None, None
            
            layer_idx = splits[layers_idx + 1]      # 例如 "0"
            module_type = splits[layers_idx + 2]    # 例如 "self_attn" 或 "mlp"
            proj_name = splits[layers_idx + 3]      # 例如 "q_proj", "down_proj"
            suffix = splits[-1]                     # "qweight" 或 "scales"
            
            # 验证 suffix 有效性
            if suffix not in ['qweight', 'scales']:
                return None, None
            
            # 构造 MNN 兼容键: "0.self_attn.q_proj"
            prefix = f"{layer_idx}.{module_type}.{proj_name}"
            return prefix, suffix
        except Exception:
            return None, None

    def get(self, key : str):
        if key in self.weight_dict:
            return self.weight_dict[key]
        return None

    def load_safetensor(self, tensor):
        self.weight_dict = dict()
        with safe_open(tensor, framework="pt") as f:
            for k in f.keys():
                p, s = self.prefix(k)
                if p is None: continue
                if s not in ['qweight', 'scales']: continue
                if p not in self.weight_dict:
                    self.weight_dict[p] = GPTQWeight(p)
                self.weight_dict[p].add(s, f.get_tensor(k))

    @staticmethod
    def weight_reorder(qweight, bits=4, group_size=128):
        oc = qweight.shape[-1]
        wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
        weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2 ** bits) - 1, out=weight)
        weight = weight.reshape(-1, oc).transpose(1, 0)
        if bits == 8:
            weight = weight.to(torch.uint8)
            return weight
        weight = weight.reshape(-1, 2).to(torch.uint8)
        weight = weight[:, 0] * 16 + weight[:, 1]
        return weight

    def apply(self, graph_path, weight_path):
        # parse mnn graph
        mnn_weights = []
        mnn_graph = json.load(open(graph_path, 'rt'))
        for op in mnn_graph['oplists']:
            if op['type'] == 'Convolution':
                name = op['name']
                external = op['main']['external']
                weight_elements = op['main']['common']['outputCount'] * op['main']['common']['inputCount']
                mnn_weights.append(MNNWeight(name, external, weight_elements))
        # load mnn weight
        external_weight = open(weight_path, 'r+b')
        for mnn_weight in mnn_weights:
            gptq_weight = self.get(mnn_weight.key())
            if gptq_weight is None: continue
            # print(f'write {mnn_weight.key()} ... ', end='')
            weight = gptq_weight.qweight
            scale = gptq_weight.scales.float().transpose(1, 0)
            # write weight data
            weight = GPTQ.weight_reorder(weight, mnn_weight.quant_bits)
            weight_bytes = weight.numpy().tobytes()
            weight_size = mnn_weight.weight_size()
            header_len = weight_size - len(weight_bytes)
            assert(header_len > 0)
            external_weight.seek(mnn_weight.offset() + header_len)
            external_weight.write(weight_bytes)
            scale_size = mnn_weight.scale_size()
            is_asy = scale.numel() * scale.element_size() < scale_size
            # write scale data
            if is_asy:
                # zeros = mnn_weight.a_min * scale
                zeros = torch.zeros_like(scale)
                scale = torch.stack([zeros, scale], axis=-1)
            scale_bytes = scale.numpy().tobytes()
            assert(scale_size == len(scale_bytes))
            external_weight.write(scale_bytes)
            # print('Done!')
        external_weight.close()


class VisualMNNWeight:
    """Represents a visual model Convolution op in the MNN graph with a known semantic role."""
    def __init__(self, name, external, weight_elements, layer_id, op_type):
        self.name = name
        self.external = external
        # Determine quant bits from weight size ratio
        self.quant_bits = 4
        if external[1] > 0 and round(weight_elements / external[1]) == 2:
            self.quant_bits = 4
            self.a_min = -8
        else:
            self.quant_bits = 8
            self.a_min = -128
        self.layer_id = str(layer_id)
        self.op_type = op_type  # e.g. "attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc1", "mlp.fc2"

    def __repr__(self) -> str:
        return f'VisualMNNWeight({self.layer_id}.{self.op_type}, {self.name})'

    def key(self):
        return f'{self.layer_id}.{self.op_type}'

    def offset(self): return self.external[0]
    def weight_size(self): return self.external[1]
    def scale_size(self): return self.external[2]


class VisualGPTQ(GPTQ):
    """Handles GPTQ weight replacement for visual (ViT) models.

    Visual model MNN Convolution op names (from onnx2mnn with transformerFuse):
      - q/k/v:  /Add_{N}_output_0__matmul_converted  (3 per block, in order)
      - o_proj:  /proj_{idx}/Add_output_0__matmul_converted  (idx omitted for block 0)
      - fc1:     /mlp/linear_fc1_{idx}/Add_output_0__matmul_converted
      - fc2:     /mlp/linear_fc2_{idx}/Add_output_0__matmul_converted
      - merger, deepstack_merger, patch_embed: non-block ops

    GPTQ safetensor keys (typical Qwen3VL visual):
      - visual.blocks.{i}.attn.qkv.qweight/scales   (fused qkv)
      - visual.blocks.{i}.attn.proj.qweight/scales
      - visual.blocks.{i}.mlp.fc1.qweight/scales
      - visual.blocks.{i}.mlp.fc2.qweight/scales
    """

    def prefix(self, name):
        """Parse visual GPTQ safetensor key into (dict_key, suffix).

        Examples:
          visual.blocks.0.attn.qkv.qweight  -> "0.attn.qkv", "qweight"
          model.visual.blocks.0.mlp.fc1.scales -> "0.mlp.fc1", "scales"
        """
        splits = name.split('.')
        suffix = splits[-1]
        if suffix not in ['qweight', 'scales']:
            return None, None
        try:
            blocks_idx = None
            for i, part in enumerate(splits):
                if part == 'blocks':
                    blocks_idx = i
                    break
            if blocks_idx is None or blocks_idx + 2 >= len(splits):
                return None, None
            layer_idx = splits[blocks_idx + 1]
            module_parts = splits[blocks_idx + 2:-1]  # e.g. ["attn", "qkv"]
            prefix = f"{layer_idx}.{'.'.join(module_parts)}"
            return prefix, suffix
        except Exception:
            return None, None

    @staticmethod
    def classify_visual_conv_ops(mnn_graph):
        """Classify visual Convolution ops into semantic roles by analyzing naming patterns.

        Returns a list of VisualMNNWeight with correct layer_id and op_type.
        Uses the naming convention from mnnconvert's transformerFuse output:
          - /proj/ or /proj_{n}/  -> o_proj for block n (0-indexed, block 0 has no suffix)
          - /mlp/linear_fc1/ or /mlp/linear_fc1_{n}/  -> fc1 for block n
          - /mlp/linear_fc2/ or /mlp/linear_fc2_{n}/  -> fc2 for block n
          - /Add_{N}_output_0__matmul_converted -> q/k/v (3 consecutive per block)
          - /patch_embed/, /merger/, /deepstack_merger_list -> skipped
        """
        import re
        results = []
        # Collect all Add_N matmul ops in order of N
        add_ops = []  # list of (N, op)
        # Collect proj/fc ops
        proj_ops = {}  # block_idx -> op
        fc1_ops = {}   # block_idx -> op
        fc2_ops = {}   # block_idx -> op

        for op in mnn_graph['oplists']:
            if op['type'] != 'Convolution':
                continue
            name = op['name']
            # Skip non-block ops
            if '/patch_embed/' in name or '/merger/' in name or '/deepstack_merger' in name:
                continue

            # Match /proj/ or /proj_{n}/
            m = re.match(r'^/proj(?:_(\d+))?/', name)
            if m:
                idx = int(m.group(1)) if m.group(1) else 0
                proj_ops[idx] = op
                continue

            # Match /mlp/linear_fc1/ or /mlp/linear_fc1_{n}/
            m = re.match(r'^/mlp/linear_fc1(?:_(\d+))?/', name)
            if m:
                idx = int(m.group(1)) if m.group(1) else 0
                fc1_ops[idx] = op
                continue

            # Match /mlp/linear_fc2/ or /mlp/linear_fc2_{n}/
            m = re.match(r'^/mlp/linear_fc2(?:_(\d+))?/', name)
            if m:
                idx = int(m.group(1)) if m.group(1) else 0
                fc2_ops[idx] = op
                continue

            # Match /Add_{N}_output_0__matmul_converted
            m = re.match(r'^/Add_(\d+)_output_0__matmul_converted$', name)
            if m:
                add_n = int(m.group(1))
                add_ops.append((add_n, op))
                continue

        # Sort Add ops by N value
        add_ops.sort(key=lambda x: x[0])

        # Group Add ops into blocks of 3 (q, k, v)
        # Each block has exactly 3 Add matmul ops for q, k, v
        for group_idx in range(len(add_ops) // 3):
            block_idx = group_idx
            for qkv_offset, qkv_name in enumerate(['attn.q', 'attn.k', 'attn.v']):
                _, op = add_ops[group_idx * 3 + qkv_offset]
                external = op['main']['external']
                ic = op['main']['common']['inputCount']
                oc = op['main']['common']['outputCount']
                results.append(VisualMNNWeight(op['name'], external, ic * oc, block_idx, qkv_name))

        # Add proj ops
        for block_idx, op in sorted(proj_ops.items()):
            external = op['main']['external']
            ic = op['main']['common']['inputCount']
            oc = op['main']['common']['outputCount']
            results.append(VisualMNNWeight(op['name'], external, ic * oc, block_idx, 'attn.proj'))

        # Add fc1 ops (key must match GPTQ safetensor: mlp.linear_fc1)
        for block_idx, op in sorted(fc1_ops.items()):
            external = op['main']['external']
            ic = op['main']['common']['inputCount']
            oc = op['main']['common']['outputCount']
            results.append(VisualMNNWeight(op['name'], external, ic * oc, block_idx, 'mlp.linear_fc1'))

        # Add fc2 ops (key must match GPTQ safetensor: mlp.linear_fc2)
        for block_idx, op in sorted(fc2_ops.items()):
            external = op['main']['external']
            ic = op['main']['common']['inputCount']
            oc = op['main']['common']['outputCount']
            results.append(VisualMNNWeight(op['name'], external, ic * oc, block_idx, 'mlp.linear_fc2'))

        return results

    @staticmethod
    def _build_gptq_weight_data(gptq_weight, quant_bits=8):
        """Build quantized weight bytes + scale bytes from a GPTQWeight.

        Supports both int8 (quant_bits=8) and int4 (quant_bits=4).
        Returns (header_bytes, weight_bytes, scale_bytes, oc, ic, shape_int32).
        """
        weight = gptq_weight.qweight
        scale = gptq_weight.scales.float().transpose(1, 0)  # (oc, num_groups)
        # Compute oc, ic from raw qweight before reorder (reorder may return 1D for int4)
        packed_ic, oc = weight.shape[0], weight.shape[-1]
        ic = packed_ic * 32 // quant_bits
        weight = GPTQ.weight_reorder(weight, quant_bits)     # int8: (oc, ic), int4: (oc*ic/2,)

        # Build header (same as MNNConverter.write_header)
        shape_dtype = np.int16
        if oc > 65535 or ic > 65535:
            shape_dtype = np.int32
        dim_bytes = b'\x02'
        shape_bytes = np.array([oc, ic]).astype(shape_dtype).tobytes()
        offset = 1 << (quant_bits - 1)  # 128 for 8-bit
        weight_map = [i for i in range(-offset, offset)]
        if len(weight_map) == 256:
            weight_map.insert(0, 0)
        else:
            weight_map.insert(0, len(weight_map))
        map_bytes = np.array(weight_map, dtype=np.int8).tobytes()
        header_bytes = dim_bytes + shape_bytes + map_bytes

        weight_bytes = weight.numpy().tobytes()
        # asymmetric: prepend zeros to scale -> (oc, num_groups, 2)
        zeros = torch.zeros_like(scale)
        scale_with_zeros = torch.stack([zeros, scale], dim=-1)
        scale_bytes = scale_with_zeros.numpy().tobytes()

        return header_bytes, weight_bytes, scale_bytes, oc, ic, shape_dtype == np.int32

    def apply(self, graph_path, weight_path, quant_block=128):
        """Replace visual model block weights with GPTQ quantized weights, keep merger/deepstack as fp16.

        Auto-detects both:
          - quant_bits (4 or 8) from qweight shape: bits = 32 * qweight.shape[0] / ic
          - num_groups (per-channel=1, per-group=ic/block_size) from scales.shape[0]

        So supports int4/int8, per-channel/per-group without user-side configuration.
        Rebuilds the entire weight file: reads all ops' external data from the original
        fp16 weight file, then writes a new weight file where block Convolutions are
        converted to quantized+scale format and non-block ops are copied as-is.
        Also updates the JSON graph's quanParameter and external fields.

        quant_block parameter is retained for API compatibility but ignored — the
        actual group layout is dictated by the GPTQ scales shape.
        """
        mnn_graph = json.load(open(graph_path, 'rt'))
        block_weights = self.classify_visual_conv_ops(mnn_graph)

        # Build a lookup: op_name -> (VisualMNNWeight, gptq_data)
        block_gptq_map = {}
        for vw in block_weights:
            gptq_w = self.get(vw.key())
            if gptq_w is not None:
                block_gptq_map[vw.name] = (vw, gptq_w)
                continue
            # Try fused qkv
            if vw.op_type in ('attn.q', 'attn.k', 'attn.v'):
                qkv_key = f'{vw.layer_id}.attn.qkv'
                gptq_qkv = self.get(qkv_key)
                if gptq_qkv is not None:
                    qkv_map = {'attn.q': 0, 'attn.k': 1, 'attn.v': 2}
                    part_idx = qkv_map[vw.op_type]
                    oc_total = gptq_qkv.qweight.shape[-1]
                    oc_part = oc_total // 3
                    part_gptq = GPTQWeight(vw.key())
                    part_gptq.qweight = gptq_qkv.qweight[:, part_idx * oc_part : (part_idx + 1) * oc_part].contiguous()
                    part_gptq.scales = gptq_qkv.scales[:, part_idx * oc_part : (part_idx + 1) * oc_part].contiguous()
                    block_gptq_map[vw.name] = (vw, part_gptq)

        # Read original fp16 weight file
        with open(weight_path, 'rb') as f:
            orig_weight_data = f.read()

        # Rebuild weight file + update JSON
        new_weight_offset = 0
        new_weight_file = open(weight_path, 'wb')
        replaced_count = 0

        for op in mnn_graph['oplists']:
            if op['type'] != 'Convolution':
                continue
            if 'external' not in op['main']:
                continue

            name = op['name']
            external = op['main']['external']
            old_offset = external[0]
            old_weight_size = external[1]
            old_scale_size = external[2]
            old_bias_size = external[3]

            ic = op['main']['common']['inputCount']
            oc = op['main']['common']['outputCount']

            if name in block_gptq_map:
                # This is a block Convolution -> write GPTQ quantized weight
                vw, gptq_w = block_gptq_map[name]
                # Auto-detect quant_bits from GPTQ qweight shape:
                #   qweight.shape[0] = ic * bits / 32, so bits = 32 * qweight.shape[0] / ic
                packed_ic = gptq_w.qweight.shape[0]
                quant_bits = 32 * packed_ic // ic
                assert quant_bits in (4, 8), f"Unexpected quant_bits={quant_bits} for {name}"
                header_bytes, weight_bytes, scale_bytes, _, _, shape_int32 = \
                    self._build_gptq_weight_data(gptq_w, quant_bits=quant_bits)

                new_weight_len = len(header_bytes) + len(weight_bytes)
                new_scale_len = len(scale_bytes)

                # Write header + weight + scale
                new_weight_file.write(header_bytes)
                new_weight_file.write(weight_bytes)
                new_weight_file.write(scale_bytes)

                # Write bias (copy from original)
                if old_bias_size > 0:
                    bias_offset = old_offset + old_weight_size + old_scale_size
                    new_weight_file.write(orig_weight_data[bias_offset : bias_offset + old_bias_size])

                # Update external
                external[0] = new_weight_offset
                external[1] = new_weight_len
                external[2] = new_scale_len
                # external[3] (bias_size) unchanged

                # Update quanParameter: fp16(type=3) -> quantized(type=1)
                # Auto-detect num_groups from GPTQ scales shape (per-channel=1, per-group=ic/block_size)
                num_groups = gptq_w.scales.shape[0]
                op['main']['quanParameter'] = {
                    'quantScale': 1.0, 'scaleIn': 0.0, 'scaleOut': 0.0,
                    'useInt32': False, 'has_scaleInt': False, 'shapeInt32': shape_int32,
                    'type': 1, 'aMaxOrBits': quant_bits, 'aMin': 1,
                    'readType': oc * num_groups, 'weightSize': 0
                }

                new_weight_offset += new_weight_len + new_scale_len + old_bias_size
                replaced_count += 1
            else:
                # Non-block Convolution (merger/deepstack/patch_embed) -> copy fp16 as-is
                total_old_size = old_weight_size + old_scale_size + old_bias_size
                new_weight_file.write(orig_weight_data[old_offset : old_offset + total_old_size])

                # Update external offset only
                external[0] = new_weight_offset
                # weight_size, scale_size, bias_size unchanged

                new_weight_offset += total_old_size

        new_weight_file.close()

        # Also copy external data for non-Convolution ops (LayerNorm, Const, etc.)
        # These need a second pass because they also have external data
        # First collect all non-Conv ops with external data
        non_conv_externals = []
        for op in mnn_graph['oplists']:
            if op['type'] == 'Convolution':
                continue
            if op['type'] == 'LayerNorm' and 'external' in op['main']:
                non_conv_externals.append(('LayerNorm', op))
            elif op['type'] == 'Const' and 'external' in op['main']:
                non_conv_externals.append(('Const', op))

        if non_conv_externals:
            with open(weight_path, 'ab') as f:
                for op_type, op in non_conv_externals:
                    ext = op['main']['external']
                    old_off = ext[0]
                    if op_type == 'LayerNorm':
                        total = ext[1] + ext[2]
                    else:
                        total = ext[1]
                    f.write(orig_weight_data[old_off : old_off + total])
                    ext[0] = new_weight_offset
                    new_weight_offset += total

        # Save updated JSON
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(mnn_graph, f, ensure_ascii=False, indent=4)

        print(f'  Visual GPTQ: replaced {replaced_count}/{len(block_weights)} block weights '
              f'(auto-detected quant bits from GPTQ weights), '
              f'kept merger/deepstack/patch_embed as fp16')