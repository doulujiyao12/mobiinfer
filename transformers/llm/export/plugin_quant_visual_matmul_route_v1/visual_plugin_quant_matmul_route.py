#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxsim import simplify as onnxsim_simplify

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXPORT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, EXPORT_DIR)


def _ensure_yaspin_stub():
    try:
        import yaspin  # noqa: F401
        return
    except ImportError:
        mod = types.ModuleType("yaspin")

        class _DummyYaspin:
            def __init__(self, text="", color=None):
                self.text = text
                self.color = color

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def hide(self):
                return None

            def show(self):
                return None

            def fail(self, *_args, **_kwargs):
                return None

            def ok(self, *_args, **_kwargs):
                return None

        mod.yaspin = _DummyYaspin
        sys.modules["yaspin"] = mod


_ensure_yaspin_stub()

try:
    from dopt.dopt_lm.do_opt import (  # noqa: E402
        generate_config_file,
        generate_quant_params,
        optimize_model,
        set_calibrate_state,
        set_quant_state,
    )
    DOPT_IMPORT_ERROR = None
except ModuleNotFoundError as import_error:
    # Pure FP16 export does not need DOPT. Keep the quantized commands available
    # when the caller configures the DDK Python path, but do not make DOPT a
    # dependency of the raw HuggingFace -> FP16 ONNX route.
    DOPT_IMPORT_ERROR = import_error
from packaging.version import Version  # noqa: E402
from utils.model import LlmModel  # noqa: E402
from utils.transformers import repeat_kv, rotate_half  # noqa: E402


SIZE_1MB = 1024 * 1024
COMPRESS_NODE_TYPES = ["Conv", "Gemm", "MatMul"]
_CONST_VAL = 0.01

DEFAULT_MODEL_PATH = "/temp/models/mobi0402_2B_halfimage_rl"
DEFAULT_INPUT_DIR = (
    "/data/dahu/mlsys/MNN/transformers/llm/export/discarded_omg_convert/"
    "pytorch_visual_omc6_route_v1/model_visual_omc6_calib/calib_inputs"
)
DEFAULT_ROUTE_DIR = os.path.join(SCRIPT_DIR, "model_visual_plugin_matmul_chunk0")


class DummyArgs:
    pass


def require_dopt():
    if DOPT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "This command requires the DDK DOPT Python package; configure "
            "PYTHONPATH for tools_dopt/dopt_pytorch_py3 first"
        ) from DOPT_IMPORT_ERROR


def make_dummy_args(model_path, dst_path):
    args = DummyArgs()
    args.tokenizer_path = model_path
    args.quant_bit = 4
    args.quant_block = 128
    args.lm_quant_bit = 16
    args.lm_quant_block = 128
    args.seperate_embed = False
    args.omni = False
    args.awq = False
    args.smooth = False
    args.export = "onnx"
    args.dst_path = dst_path
    args.transformer_fuse = False
    args.group_conv_native = False
    args.visual_quant_bit = 4
    args.visual_quant_block = 128
    args.visual_sym = False
    args.sym = False
    args.hqq = False
    args.visual_keep_matmul = False
    args.visual_gptq_path = None
    args.tie_word_embeddings = False
    args.onnx_slim = False
    args.keep_onnx = True
    args.cleanup_onnx = False
    args.lora_path = None
    args.lora_split = False
    args.eagle_path = None
    args.embed_bit = 16
    args.skip_weight = False
    args.act_bit = 16
    args.act_sym = False
    args.generate_for_npu = False
    args.ppl = False
    args.test = None
    args.quant_config = None
    args.visual_split = True
    args.visual_npu_chunks = 0
    args.visual_npu_layers = 0
    args.visual_chunk_backends = ""
    args.visual_json_full = False
    args.visual_no_json = False
    args.calib_data = None
    args.omni_epochs = 20
    args.omni_lr = 5e-3
    args.omni_wd = 1e-4
    return args


def _to_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Unsupported type: {type(obj)}")


def _del_nodes(graph, nodes):
    idxs = sorted([i for i, n in enumerate(graph.node) if n in nodes], reverse=True)
    for i in idxs:
        del graph.node[i]


def _del_inits(graph, names):
    idxs = sorted([i for i, t in enumerate(graph.initializer) if t.name in names], reverse=True)
    for i in idxs:
        del graph.initializer[i]


def _make_const_of_shape(shape, output_name):
    name = f"csz_{output_name}"
    sinit = onnx.helper.make_tensor(name + "_shape", onnx.TensorProto.INT64, [len(shape)], shape)
    val = onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT, [1], [float(_CONST_VAL)])
    node = onnx.helper.make_node("ConstantOfShape", inputs=[sinit.name], outputs=[output_name], value=val)
    return node, sinit


def _compress_onnx(onnx_model):
    graph = onnx_model.graph
    name2init = {i.name: i for i in graph.initializer}
    new_nodes, new_inits, removed = [], [], []
    for node in graph.node:
        if node.op_type not in COMPRESS_NODE_TYPES:
            continue
        w = node.input[1] if len(node.input) > 1 else None
        if not w or w not in name2init:
            continue
        init = name2init[w]
        if init.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
            continue
        shape = [d for d in init.dims]
        dtype_bytes = {onnx.TensorProto.FLOAT: 4, onnx.TensorProto.FLOAT16: 2}[init.data_type]
        if int(np.prod(shape)) * dtype_bytes <= SIZE_1MB:
            continue
        global _CONST_VAL
        cnode, sinit = _make_const_of_shape(shape, w)
        _CONST_VAL += 0.003
        removed.append(init)
        new_nodes.append(cnode)
        new_inits.append(sinit)
    _del_inits(graph, [i.name for i in removed])
    for node in reversed(new_nodes):
        graph.node.insert(0, node)
    graph.initializer.extend(new_inits)
    return onnx_model, removed


def _uncompress_onnx(onnx_model, removed):
    onnx_model.graph.initializer.extend(removed)
    replaced = set(i.name for i in removed)
    del_nodes = [
        n for n in onnx_model.graph.node if n.op_type == "ConstantOfShape" and n.output[0] in replaced
    ]
    _del_nodes(onnx_model.graph, del_nodes)
    return onnx_model


def _fix_onnx_names(onnx_model):
    used = set()
    for node in onnx_model.graph.node:
        if node.op_type == "Gather" and node.name and node.input[0].endswith(".weight"):
            node.name = node.input[0][:-7]
            continue
        if node.op_type in ("MatMul", "Gemm", "Mul", "Cast", "Transpose"):
            name = node.name
            if not name or "/" not in name:
                continue
            if name.startswith("/"):
                name = name[1:]
            parts = name.split("/")
            if len(parts) <= 1:
                continue
            new_name = ".".join(parts[:-1])
            if new_name in used:
                idx = 1
                while f"{new_name}_{idx}" in used:
                    idx += 1
                new_name = f"{new_name}_{idx}"
            used.add(new_name)
            node.name = new_name
    return onnx_model


def _split_shared_weights(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph
    counts = {}
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        w = node.input[1]
        counts[w] = counts.get(w, 0) + 1
        if counts[w] <= 1:
            continue
        for init in graph.initializer:
            if init.name == w:
                new_name = f"{w}_{counts[w]}"
                node.input[1] = new_name
                new_init = copy.deepcopy(init)
                new_init.name = new_name
                graph.initializer.append(new_init)
                break
    shared = {k: v for k, v in counts.items() if v > 1}
    if shared:
        print(f"  split_weights: {sum(v - 1 for v in shared.values())}")

    graph2 = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
    info = onnx.helper.make_model(graph2)
    info.opset_import[0].version = 12

    new_onnx = os.path.realpath(onnx_path)[:-5] + ".onnx"
    pb_name = os.path.basename(new_onnx)[:-5] + ".pb"
    os.unlink(onnx_path)
    if os.path.exists(pb_name):
        os.remove(pb_name)
    onnx.save(
        info,
        new_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=pb_name,
        convert_attribute=False,
    )
    return new_onnx


def process_onnx_for_omg(onnx_path, fp16=False):
    model = onnx.load(onnx_path)
    matmul_weights = set()
    if fp16:
        for node in model.graph.node:
            if node.op_type in ("MatMul", "Gemm", "Conv") and len(node.input) >= 2:
                matmul_weights.add(node.input[1])

    model, removed = _compress_onnx(model)
    print(f"  compress: {len(removed)} large weights")

    skipped = ["fuse_matmul_add_bias_into_gemm", "fuse_qkv", "eliminate_duplicate_initializer"]
    model, check = onnxsim_simplify(model, tensor_size_threshold="1024KB", skipped_optimizers=skipped)
    print(f"  onnxsim: {check}")

    model = _uncompress_onnx(model, removed)
    print("  uncompress: done")

    if fp16:
        converted = 0
        for init in model.graph.initializer:
            if init.name not in matmul_weights:
                continue
            if init.data_type != onnx.TensorProto.FLOAT:
                continue
            data = onnx.numpy_helper.to_array(init)
            init.data_type = onnx.TensorProto.FLOAT16
            init.raw_data = data.astype(np.float16).tobytes()
            for field in ("float_data", "int32_data"):
                try:
                    init.ClearField(field)
                except ValueError:
                    pass
            converted += 1
        print(f"  fp16: {converted} matmul weights converted")

    model = _fix_onnx_names(model)
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            inp.type.tensor_type.elem_type = onnx.TensorProto.INT32

    os.unlink(onnx_path)
    pb = os.path.basename(onnx_path) + ".pb"
    onnx.save(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=pb,
        convert_attribute=False,
    )
    return _split_shared_weights(onnx_path)


def build_chunk_specs(total_blocks, npu_chunks):
    base, rem = total_blocks // npu_chunks, total_blocks % npu_chunks
    specs = []
    cursor = 0
    for ci in range(npu_chunks):
        size = base + (1 if ci < rem else 0)
        specs.append((cursor, cursor + size, ci))
        cursor += size
    return specs


class _FLinearMatmul(torch.nn.Module):
    def __init__(self, m: torch.nn.Linear, algin_dim=1, squeeze_batch=False, unsqueeze_before_bias=False):
        super().__init__()
        self.in_features = m.in_features
        self.out_features = m.out_features
        self.squeeze_batch = squeeze_batch
        self.unsqueeze_before_bias = unsqueeze_before_bias

        if m.out_features % algin_dim != 0:
            self.out_features = (m.out_features + algin_dim - 1) // algin_dim * algin_dim
            weight = torch.zeros((self.out_features, self.in_features), dtype=m.weight.dtype, device=m.weight.device)
            weight[:m.out_features, :] = m.weight.data
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            if m.bias is not None:
                bias = torch.zeros(self.out_features, dtype=m.bias.dtype, device=m.bias.device)
                bias[:m.out_features] = m.bias.data
                self.bias = torch.nn.Parameter(bias, requires_grad=False)
            else:
                self.bias = None
        else:
            self.weight = torch.nn.Parameter(m.weight.data.clone(), requires_grad=False)
            self.bias = (
                torch.nn.Parameter(m.bias.data.clone(), requires_grad=False)
                if m.bias is not None
                else None
            )

    def forward(self, x):
        if self.squeeze_batch:
            x = x.reshape(x.shape[1], x.shape[2])
        out = torch.nn.functional.linear(x, self.weight)
        if self.squeeze_batch and self.unsqueeze_before_bias:
            out = out.reshape(1, out.shape[0], out.shape[1])
        if self.bias is not None:
            out = out + self.bias
        if self.squeeze_batch and (not self.unsqueeze_before_bias):
            out = out.reshape(1, out.shape[0], out.shape[1])
        return out


def _replace_linear_with_flinear(module, algin_dim=1, squeeze_batch=False, unsqueeze_before_bias=False):
    for name, child in list(module.named_children()):
        if type(child) == torch.nn.Linear:
            setattr(
                module,
                name,
                _FLinearMatmul(
                    child,
                    algin_dim=algin_dim,
                    squeeze_batch=squeeze_batch,
                    unsqueeze_before_bias=unsqueeze_before_bias,
                ),
            )
        else:
            _replace_linear_with_flinear(
                child,
                algin_dim=algin_dim,
                squeeze_batch=squeeze_batch,
                unsqueeze_before_bias=unsqueeze_before_bias,
            )
    return module


class _ExportAttentionAdapter(torch.nn.Module):
    def __init__(self, attn, use_qwen3_style_rotary: bool = False):
        super().__init__()
        self.attn = attn
        self.use_qwen3_style_rotary = use_qwen3_style_rotary

    def _linear_3d_to_2d_to_3d(self, linear, hidden_states):
        bsz, q_len, dim = hidden_states.shape
        hidden_states_2d = hidden_states.reshape(bsz * q_len, dim)
        out_2d = linear(hidden_states_2d)
        return out_2d.reshape(bsz, q_len, -1)

    def _normalize_rotary(self, rotary_pos_emb):
        if rotary_pos_emb is None:
            return None, None
        if rotary_pos_emb.dim() == 5:
            cos, sin = rotary_pos_emb[0], rotary_pos_emb[1]
        elif rotary_pos_emb.dim() == 4:
            cos, sin = rotary_pos_emb[0], rotary_pos_emb[1]
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported rotary_pos_emb rank: {rotary_pos_emb.dim()}")
        return cos, sin

    def _normalize_rotary_qwen3(self, rotary_pos_emb, bsz: int):
        cos, sin = self._normalize_rotary(rotary_pos_emb)
        if cos is None or sin is None:
            return None, None
        # Keep the packed rotary input format unchanged, but reinterpret it using
        # the official Qwen3 broadcast layout: [B, 1, S, D].
        if cos.dim() != 4 or sin.dim() != 4:
            raise ValueError(
                f"Expected normalized rotary rank-4 tensors, got cos={cos.dim()} sin={sin.dim()}"
            )
        cos = cos.squeeze(2).unsqueeze(1)
        sin = sin.squeeze(2).unsqueeze(1)
        if cos.shape[0] == 1 and bsz != 1:
            cos = cos.expand(bsz, -1, -1, -1)
            sin = sin.expand(bsz, -1, -1, -1)
        return cos, sin

    def _apply_qwen3_style_rotary(self, x, cos, sin):
        return (x * cos) + (rotate_half(x) * sin)

    def forward(self, hidden_states, attention_mask=None, rotary_pos_emb=None):
        attn = self.attn
        bsz, q_len, _ = hidden_states.size()
        query_states = self._linear_3d_to_2d_to_3d(attn.q_proj, hidden_states)
        if attn.q_proj.out_features == 2 * attn.num_heads * attn.head_dim:
            reshaped = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim * 2)
            query_states, gate = torch.split(reshaped, attn.head_dim, dim=-1)
            gate = gate.reshape(bsz, q_len, -1)
        else:
            gate = None

        query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            query_states = attn.q_norm(query_states)

        shared_kv_cache = getattr(attn, "_shared_kv_cache", None)
        use_shared_kv = (
            attn.is_kv_shared_layer
            and shared_kv_cache is not None
            and attn.kv_shared_layer_index in shared_kv_cache
            and not torch.onnx.is_in_onnx_export()
        )

        if use_shared_kv:
            key_states, value_states = shared_kv_cache[attn.kv_shared_layer_index]
        elif attn.k_proj is not None:
            key_states = self._linear_3d_to_2d_to_3d(attn.k_proj, hidden_states)
            if attn.k_eq_v:
                value_states = key_states.clone()
            else:
                value_states = self._linear_3d_to_2d_to_3d(attn.v_proj, hidden_states)
            key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim)
            value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim)
            if hasattr(attn, "k_norm") and attn.k_norm is not None:
                key_states = attn.k_norm(key_states)
            if hasattr(attn, "v_norm") and attn.v_norm is not None:
                value_states = attn.v_norm(value_states)
        else:
            key_states = query_states.new_zeros(bsz, q_len, attn.num_key_value_heads, attn.head_dim)
            value_states = key_states

        use_qwen3_style_rotary = (
            self.use_qwen3_style_rotary
            and attn.rotary is not None
            and attn.past_key_value is None
            and not use_shared_kv
        )

        if use_qwen3_style_rotary:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            cos, sin = self._normalize_rotary_qwen3(rotary_pos_emb, bsz)
            query_states = self._apply_qwen3_style_rotary(query_states, cos, sin)
            if attn.k_proj is not None:
                key_states = self._apply_qwen3_style_rotary(key_states, cos, sin)
        elif attn.rotary is not None:
            cos, sin = self._normalize_rotary(rotary_pos_emb)
            query_states = attn.rotary.apply_rotary_pos(query_states, cos, sin)
            if not use_shared_kv and attn.k_proj is not None:
                key_states = attn.rotary.apply_rotary_pos(key_states, cos, sin)

        if hasattr(attn, "qk_norm") and attn.qk_norm is not None:
            query_states = attn.qk_norm(query_states)
            key_states = attn.qk_norm(key_states)

        if attn.export_fused_attn and torch.onnx.is_in_onnx_export():
            if use_qwen3_style_rotary:
                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
            attn_output = attn.fused_attn(query_states, key_states, value_states, attention_mask)
            if gate is not None:
                attn_output = attn_output * torch.sigmoid(gate)
            return self._linear_3d_to_2d_to_3d(attn.o_proj, attn_output)

        if attn.past_key_value is not None:
            past_key, past_value = attn.past_key_value[0], attn.past_key_value[1]
            key_states = torch.cat((past_key, key_states), dim=1)
            value_states = torch.cat((past_value, value_states), dim=1)

        if not use_shared_kv:
            if use_qwen3_style_rotary:
                # Keep KV cache layout compatible with the original adapter path:
                # [B, S, H, D] for both key and value.
                cache_key_states = key_states.transpose(1, 2)
                cache_value_states = value_states
                attn.past_key_value = torch.stack((cache_key_states, cache_value_states))
            else:
                attn.past_key_value = torch.stack((key_states, value_states))

        if use_qwen3_style_rotary:
            if not use_shared_kv:
                key_states = key_states.permute([0, 1, 3, 2])
                value_states = value_states.transpose(1, 2)
                if attn.store_full_length_kv and shared_kv_cache is not None:
                    shared_kv_cache[attn.layer_id] = (key_states.clone(), value_states.clone())
        else:
            query_states = query_states.transpose(1, 2)
            if not use_shared_kv:
                key_states = key_states.permute([0, 2, 3, 1])
                value_states = value_states.transpose(1, 2)
                if attn.store_full_length_kv and shared_kv_cache is not None:
                    shared_kv_cache[attn.layer_id] = (key_states.clone(), value_states.clone())

        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)
        attn_scaling = getattr(attn, "attn_scaling", 1.0 / math.sqrt(attn.head_dim))
        attn_weights = torch.matmul(query_states, key_states) * attn_scaling
        if attention_mask.dtype in (torch.bool, torch.int32):
            attn_weights.masked_fill_(attention_mask, -10000.0)
        else:
            attn_weights = attn_weights + attention_mask

        if hasattr(attn, "sinks"):
            sinks = attn.sinks.reshape(1, -1, 1, 1).to(torch.float32).expand(
                query_states.shape[0], -1, query_states.shape[-2], -1
            )
            combined_logits = torch.cat([attn_weights, sinks], dim=-1)
            combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
            probs = torch.nn.functional.softmax(combined_logits, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_weights = probs[..., :-1]
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        return self._linear_3d_to_2d_to_3d(attn.o_proj, attn_output)


class _ExportMlpAdapter(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def _linear_3d_to_2d_to_3d(self, linear, hidden_states):
        bsz, q_len, dim = hidden_states.shape
        hidden_states_2d = hidden_states.reshape(bsz * q_len, dim)
        out_2d = linear(hidden_states_2d)
        return out_2d.reshape(bsz, q_len, -1)

    def forward(self, hidden_states):
        mlp = self.mlp
        if hasattr(mlp, "linear_fc1") and hasattr(mlp, "linear_fc2"):
            hidden_states = self._linear_3d_to_2d_to_3d(mlp.linear_fc1, hidden_states)
            hidden_states = mlp.act_fn(hidden_states)
            hidden_states = self._linear_3d_to_2d_to_3d(mlp.linear_fc2, hidden_states)
            return hidden_states
        if all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
            gate = self._linear_3d_to_2d_to_3d(mlp.gate_proj, hidden_states)
            up = self._linear_3d_to_2d_to_3d(mlp.up_proj, hidden_states)
            hidden_states = mlp.act_fn(gate) * up
            return self._linear_3d_to_2d_to_3d(mlp.down_proj, hidden_states)
        return mlp(hidden_states)


def _prepare_export_block(block, use_qwen3_style_rotary: bool = False):
    if getattr(block, "_plugin_quant_matmul_patched", False):
        return block
    if hasattr(block, "self_attn") and block.self_attn is not None:
        attn_impl = block.self_attn.attn if hasattr(block.self_attn, "attn") else block.self_attn
        _replace_linear_with_flinear(attn_impl)
        if not isinstance(block.self_attn, _ExportAttentionAdapter):
            block.self_attn = _ExportAttentionAdapter(
                attn_impl, use_qwen3_style_rotary=use_qwen3_style_rotary
            )
    if hasattr(block, "mlp") and block.mlp is not None:
        mlp_impl = block.mlp.mlp if hasattr(block.mlp, "mlp") else block.mlp
        _replace_linear_with_flinear(mlp_impl)
        if not isinstance(block.mlp, _ExportMlpAdapter):
            block.mlp = _ExportMlpAdapter(mlp_impl)
    block._plugin_quant_matmul_patched = True
    return block


class VisualBlocksChunkNoDS(torch.nn.Module):
    def __init__(self, visual, start, end):
        super().__init__()
        self.blocks = torch.nn.ModuleList(list(visual.blocks[start:end]))

    def forward(self, hidden_states, rotary_pos_emb, attention_mask):
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
            )
        return hidden_states


class VisualBlocksChunkDS(torch.nn.Module):
    def __init__(self, visual, start, end, local_ds):
        super().__init__()
        self.blocks = torch.nn.ModuleList(list(visual.blocks[start:end]))
        self._local_ds = list(local_ds)

    def forward(self, hidden_states, rotary_pos_emb, attention_mask):
        ds = []
        for idx, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
            )
            if idx in self._local_ds:
                ds.append(hidden_states)
        return (hidden_states, *ds)


def make_chunk_module(visual, start, end):
    has_deepstack = (
        hasattr(visual, "deepstack_visual_indexes")
        and visual.deepstack_visual_indexes is not None
        and len(visual.deepstack_visual_indexes) > 0
    )
    ds_global = list(visual.deepstack_visual_indexes) if has_deepstack else []
    local_ds = [i - start for i in ds_global if start <= i < end]
    if local_ds:
        return VisualBlocksChunkDS(visual, start, end, local_ds), local_ds
    return VisualBlocksChunkNoDS(visual, start, end), local_ds


def sanitize_chunk_for_plugin_quant(chunk):
    for blk in getattr(chunk, "blocks", []):
        attn_adapter = getattr(blk, "self_attn", None)
        if attn_adapter is not None and hasattr(attn_adapter, "config") and isinstance(attn_adapter.config, torch.nn.Module):
            attn_adapter.config = None
        attn = getattr(attn_adapter, "attn", None)
        if attn is not None and hasattr(attn, "config") and isinstance(attn.config, torch.nn.Module):
            attn.config = None
    return chunk


def reset_kv_cache(module: torch.nn.Module):
    for sub in module.modules():
        if hasattr(sub, "past_key_value"):
            sub.past_key_value = None
    return module


def load_visual_chunk(
    model_path: str,
    route_dir: str,
    npu_chunks: int,
    chunk_index: int,
    prepare_export: bool = False,
    use_qwen3_style_rotary: bool = False,
):
    args = make_dummy_args(model_path, route_dir)
    args.npu_chunks = npu_chunks
    model = LlmModel.from_pretrained(model_path, args=args)
    visual = model.visual
    specs = build_chunk_specs(len(visual.blocks), npu_chunks)
    if chunk_index < 0 or chunk_index >= len(specs):
        raise IndexError(f"chunk_index {chunk_index} out of range: {len(specs)}")
    block_start, block_end, ci = specs[chunk_index]
    chunk, local_ds = make_chunk_module(visual, block_start, block_end)
    sanitize_chunk_for_plugin_quant(chunk)
    if prepare_export:
        for blk in getattr(chunk, "blocks", []):
            _prepare_export_block(blk, use_qwen3_style_rotary=use_qwen3_style_rotary)
    chunk.eval()
    meta = {
        "chunk_index": ci,
        "block_start": block_start,
        "block_end": block_end - 1,
        "local_deepstack_count": len(local_ds),
        "num_total_chunks": len(specs),
    }
    return chunk, meta


def patch_quant_config(
    config_path: str,
    quant_strategy: str,
    weight_bit: int,
    group_size: int,
    emit_group_size: bool,
    weight_algo: str | None,
    act_bit: int,
    input_algo: str | None,
    unsigned_quant: bool,
    enable_output_quant: bool,
    output_bit: int,
    output_per_channel: bool,
    output_input_algo: str,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    layer_strategy = cfg.get("layer_strategy", {})
    patched = []
    skipped = []
    for name, layer in layer_strategy.items():
        layer_type = layer.get("type")
        if layer_type != "<class 'torch.nn.modules.linear.Linear'>":
            skipped.append(name)
            continue
        if name.endswith("qkv_proj"):
            skipped.append(name)
            continue
        layer["quant_strategy"] = quant_strategy
        layer["weight"] = {
            "bit": weight_bit,
        }
        if emit_group_size:
            layer["weight"]["group_size"] = group_size
        if weight_algo:
            layer["weight"]["weight_algo"] = weight_algo
        layer["input"] = {
            "bit": act_bit,
        }
        if input_algo:
            layer["input"]["input_algo"] = input_algo
        if unsigned_quant:
            layer["input"]["unsigned_quant"] = True
        if enable_output_quant:
            layer["output"] = {
                "bit": output_bit,
                "per_channel": output_per_channel,
                "input_algo": output_input_algo,
            }
        else:
            layer.pop("output", None)
        patched.append(name)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return {
        "patched": patched,
        "skipped": skipped,
        "total_layers": len(layer_strategy),
        "quant_strategy": quant_strategy,
        "weight_bit": weight_bit,
        "group_size": group_size if emit_group_size else None,
        "emit_group_size": emit_group_size,
        "weight_algo": weight_algo,
        "act_bit": act_bit,
        "input_algo": input_algo,
        "unsigned_quant": unsigned_quant,
        "enable_output_quant": enable_output_quant,
        "output_bit": output_bit if enable_output_quant else None,
        "output_per_channel": output_per_channel if enable_output_quant else None,
        "output_input_algo": output_input_algo if enable_output_quant else None,
    }


def load_calibration_samples(
    input_dir: str,
    chunk_index: int,
    num_samples: int,
    force_fp32: bool = True,
    sample_prefix: str | None = None,
):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    prefix = sample_prefix if sample_prefix is not None else f"chunk_{chunk_index:02d}"
    files = sorted(input_path.glob(f"{prefix}_sample_*.npz"))
    if not files:
        raise FileNotFoundError(f"No calibration npz found for prefix '{prefix}' in {input_dir}")
    if num_samples > 0:
        files = files[:num_samples]
    samples = []
    manifest = []
    for file_path in files:
        data = np.load(file_path)
        sample = {
            "hidden_states_in": torch.from_numpy(data["hidden_states_in"]),
            "rotary_pos_emb": torch.from_numpy(data["rotary_pos_emb"]),
            "attention_mask": torch.from_numpy(data["attention_mask"]),
        }
        if force_fp32:
            sample = {k: v.to(torch.float32) for k, v in sample.items()}
        samples.append(sample)
        manifest.append(
            {
                "file": str(file_path),
                "shapes": {k: list(v.shape) for k, v in sample.items()},
                "dtypes": {k: str(v.dtype) for k, v in sample.items()},
            }
        )
    return samples, manifest


def ensure_route_layout(route_dir: str):
    route = Path(route_dir)
    route.mkdir(parents=True, exist_ok=True)
    for sub in ("quant_output", "onnx", "omc_output", "logs"):
        (route / sub).mkdir(parents=True, exist_ok=True)
    return route


def config_path(route_dir: str, chunk_index: int):
    return os.path.join(route_dir, f"dopt_config.chunk_{chunk_index:02d}.json")


def calibrated_state_path(route_dir: str, chunk_index: int):
    return os.path.join(route_dir, "quant_output", f"calibrated_chunk_{chunk_index:02d}.pth")


def quant_output_dir(route_dir: str):
    return os.path.join(route_dir, "quant_output")


def fake_quant_weight_path(route_dir: str):
    return os.path.join(route_dir, "quant_output", "fake_quant_weight.pth")


def quant_params_path(route_dir: str):
    return os.path.join(route_dir, "quant_output", "quant_params_file")


def remap_fake_quant_state_for_export(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = re.sub(
            r"^(blocks\.\d+\.self_attn)\.(q_proj|k_proj|v_proj|o_proj|qkv_proj)(\..+)$",
            r"\1.attn.\2\3",
            new_key,
        )
        new_key = re.sub(
            r"^(blocks\.\d+\.mlp)\.(linear_fc1|linear_fc2)(\..+)$",
            r"\1.mlp.\2\3",
            new_key,
        )
        remapped[new_key] = value
    return remapped


def build_input_shape(sample):
    h = sample["hidden_states_in"].shape
    r = sample["rotary_pos_emb"].shape
    a = sample["attention_mask"].shape
    return (
        f"hidden_states_in:{h[0]},{h[1]},{h[-1]};"
        f"rotary_pos_emb:{r[0]},{r[1]},{r[2]},{r[3]};"
        f"attention_mask:{a[0]},{a[1]},{a[2]}"
    )


def align_onnx_node_names_with_quant_params(onnx_path: str):
    return {}


def write_route_config(route_dir: str, onnx_path: str, sample, output_names, meta):
    route_cfg = {
        "model": os.path.realpath(onnx_path),
        "input_shape": build_input_shape(sample),
        "out_nodes": ",".join(output_names),
        "outputs": output_names,
        "inputs_order": ["hidden_states_in", "rotary_pos_emb", "attention_mask"],
        **meta,
    }
    route_cfg_path = os.path.join(route_dir, "route_config.json")
    with open(route_cfg_path, "w", encoding="utf-8") as f:
        json.dump(route_cfg, f, indent=2, ensure_ascii=False, default=_to_serializable)
    return route_cfg_path


def prepare_config(args):
    require_dopt()
    ensure_route_layout(args.route_dir)
    chunk, meta = load_visual_chunk(args.model_path, args.route_dir, args.npu_chunks, args.chunk_index)
    cfg_path = config_path(args.route_dir, args.chunk_index)
    if not os.path.exists(cfg_path) or args.force_regen:
        generate_config_file(chunk, cfg_path)
    patch_info = patch_quant_config(
        cfg_path,
        quant_strategy=args.quant_strategy,
        weight_bit=args.weight_bit,
        group_size=args.group_size,
        emit_group_size=not args.omit_group_size,
        weight_algo=args.weight_algo,
        act_bit=args.act_bit,
        input_algo=args.input_algo,
        unsigned_quant=args.input_unsigned_quant,
        enable_output_quant=args.enable_output_quant,
        output_bit=args.output_bit,
        output_per_channel=args.output_per_channel,
        output_input_algo=args.output_input_algo,
    )
    summary = {
        "config": cfg_path,
        "patch_info": patch_info,
        **meta,
    }
    summary_path = os.path.join(args.route_dir, "config_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_to_serializable)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_to_serializable))


def calibrate_and_export_quant(args):
    require_dopt()
    ensure_route_layout(args.route_dir)
    cfg_path = config_path(args.route_dir, args.chunk_index)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}. Run prepare first.")
    chunk, meta = load_visual_chunk(args.model_path, args.route_dir, args.npu_chunks, args.chunk_index)
    qmodel = optimize_model(chunk, cfg_path)
    qmodel.eval()
    set_quant_state(qmodel, weight_state=True, input_state=True)
    set_calibrate_state(qmodel, True)
    samples, manifest = load_calibration_samples(
        args.input_dir, args.chunk_index, args.num_samples, sample_prefix=args.sample_prefix
    )
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            reset_kv_cache(qmodel)
            _ = qmodel(
                sample["hidden_states_in"],
                sample["rotary_pos_emb"],
                sample["attention_mask"],
            )
            print(f"calibrated sample {idx}: {manifest[idx]['file']}")
    set_calibrate_state(qmodel, False)
    state_path = calibrated_state_path(args.route_dir, args.chunk_index)
    torch.save(qmodel.state_dict(), state_path)
    set_quant_state(qmodel, weight_state=True, input_state=True)
    generate_quant_params(
        qmodel,
        quant_output_dir(args.route_dir),
        quant_param_2=False,
        embedding_separate=False,
    )
    report = {
        "calibrated_state": state_path,
        "fake_quant_weight": fake_quant_weight_path(args.route_dir),
        "quant_params_file": quant_params_path(args.route_dir),
        "num_samples": len(samples),
        "samples": manifest,
        "act_bit": args.act_bit,
        "input_unsigned_quant": args.input_unsigned_quant,
        "weight_bit": args.weight_bit,
        "group_size": args.group_size,
        **meta,
    }
    report_path = os.path.join(args.route_dir, "quant_output", "calibration_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_to_serializable)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_to_serializable))


def export_chunk_onnx(args, sample, state, weight_source):
    ensure_route_layout(args.route_dir)
    chunk, meta = load_visual_chunk(
        args.model_path,
        args.route_dir,
        args.npu_chunks,
        args.chunk_index,
        prepare_export=True,
        use_qwen3_style_rotary=args.use_qwen3_style_rotary,
    )
    missing = []
    unexpected = []
    if state is not None:
        state = remap_fake_quant_state_for_export(state)
        missing, unexpected = chunk.load_state_dict(state, strict=False)
    onnx_path = os.path.join(args.route_dir, "onnx", f"visual_blocks_npu_{args.chunk_index}.onnx")
    out_names = ["hidden_states"]
    local_ds_count = meta["local_deepstack_count"]
    for idx in range(local_ds_count):
        out_names.append(f"deepstack_hidden_{idx}")
    export_kwargs = dict(
        opset_version=12,
        do_constant_folding=True,
        input_names=["hidden_states_in", "rotary_pos_emb", "attention_mask"],
        output_names=out_names,
        verbose=False,
    )
    if Version(torch.__version__) >= Version("2.4.0"):
        export_kwargs["dynamo"] = False
    with torch.no_grad():
        reset_kv_cache(chunk)
        torch.onnx.export(
            chunk,
            (
                sample["hidden_states_in"],
                sample["rotary_pos_emb"],
                sample["attention_mask"],
            ),
            onnx_path,
            **export_kwargs,
        )
    final_onnx = process_onnx_for_omg(onnx_path, fp16=args.fp16)
    rename_map = align_onnx_node_names_with_quant_params(final_onnx)
    route_cfg_path = write_route_config(args.route_dir, final_onnx, sample, out_names, meta)
    report = {
        "onnx": final_onnx,
        "route_config": route_cfg_path,
        "renamed_nodes": rename_map,
        "missing_keys": sorted(list(missing)),
        "unexpected_keys": sorted(list(unexpected)),
        "output_names": out_names,
        "weight_source": weight_source,
        "calibration_used": state is not None,
        "act_bit": args.act_bit,
        "input_unsigned_quant": args.input_unsigned_quant,
        "weight_bit": args.weight_bit,
        "group_size": args.group_size,
        **meta,
    }
    report_path = os.path.join(args.route_dir, "onnx", "export_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_to_serializable)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_to_serializable))


def export_onnx(args):
    fake_quant_path = fake_quant_weight_path(args.route_dir)
    if not os.path.exists(fake_quant_path):
        raise FileNotFoundError(
            f"Missing fake_quant_weight: {fake_quant_path}. Run calibrate first."
        )
    state = torch.load(fake_quant_path, map_location="cpu")
    samples, _manifest = load_calibration_samples(
        args.input_dir, args.chunk_index, 1, sample_prefix=args.sample_prefix
    )
    export_chunk_onnx(args, samples[0], state, "dopt_fake_quant_weight")


def export_raw_fp16_onnx(args):
    if not args.fp16:
        raise ValueError("export-fp16 requires --fp16 so MatMul/Gemm weights are stored as FLOAT16")
    if args.sequence_length <= 0 or args.hidden_size <= 0 or args.rotary_size <= 0:
        raise ValueError("sequence_length, hidden_size, and rotary_size must be positive")
    # These tensors provide static tracing shapes only. No calibration or
    # value-dependent control flow is used in the visual block exporter.
    sample = {
        "hidden_states_in": torch.zeros(
            (1, args.sequence_length, args.hidden_size), dtype=torch.float32
        ),
        "rotary_pos_emb": torch.zeros(
            (2, args.sequence_length, 1, args.rotary_size), dtype=torch.float32
        ),
        "attention_mask": torch.zeros(
            (1, args.sequence_length, args.sequence_length), dtype=torch.float32
        ),
    }
    export_chunk_onnx(args, sample, None, "original_huggingface_float")


def run_all(args):
    prepare_config(args)
    calibrate_and_export_quant(args)
    export_onnx(args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone plugin-quant + FLinearMatmul plain-MatMul export route for visual block chunks"
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="HF model path")
    parser.add_argument("--route_dir", default=DEFAULT_ROUTE_DIR, help="Experiment output directory")
    parser.add_argument("--npu_chunks", type=int, default=6, help="Total visual chunk count")
    parser.add_argument("--chunk_index", type=int, default=0, help="Chunk index to process")
    parser.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help="Calibration npz directory reused as PTQ samples",
    )
    parser.add_argument(
        "--sample_prefix",
        default=None,
        help=(
            "Calibration npz file prefix; when unset defaults to chunk_{chunk_index:02d}. "
            "Use this to switch to custom-seqlen inputs generated by collect_visual_act_stats_v6.py"
        ),
    )
    parser.add_argument("--num_samples", type=int, default=4, help="Calibration sample count")
    parser.add_argument("--fp16", action="store_true", help="Convert large MatMul/Gemm weights to fp16")
    parser.add_argument(
        "--sequence_length", type=int, default=608,
        help="Static visual sequence length for raw FP16 export",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=1024,
        help="Visual hidden size for raw FP16 export",
    )
    parser.add_argument(
        "--rotary_size", type=int, default=64,
        help="Per-head rotary size for raw FP16 export",
    )
    parser.add_argument("--force_regen", action="store_true", help="Regenerate quant config")
    parser.add_argument("--quant_strategy", default="Quant_act_weight_eco", help="Plugin-quant strategy")
    parser.add_argument("--weight_bit", type=int, default=4, help="Weight bit width")
    parser.add_argument("--group_size", type=int, default=128, help="Weight group size")
    parser.add_argument(
        "--omit_group_size",
        action="store_true",
        help="Omit weight.group_size from dopt_config; default behavior still writes it",
    )
    parser.add_argument(
        "--weight_algo",
        default=None,
        help="Optional weight quant algo; omitted from dopt_config when unset",
    )
    parser.add_argument("--act_bit", type=int, default=16, help="Activation bit width")
    parser.add_argument(
        "--input_algo",
        default=None,
        help="Optional input quant algo; omitted from dopt_config when unset",
    )
    parser.add_argument(
        "--enable_output_quant",
        action="store_true",
        help="Add output quant config for linear layers; intended for Kirin9020 experiments",
    )
    parser.add_argument("--output_bit", type=int, default=16, help="Output quant bit width")
    parser.add_argument(
        "--output_per_channel",
        type=lambda s: str(s).lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Use per-channel output quantization when output quant is enabled",
    )
    parser.add_argument(
        "--output_input_algo",
        default="min_max",
        help="Output quant input_algo when output quant is enabled",
    )
    parser.add_argument(
        "--input_unsigned_quant",
        action="store_true",
        help="Use unsigned activation quantization; default is signed for A16 experiments",
    )
    parser.add_argument(
        "--use_qwen3_style_rotary",
        action="store_true",
        help=(
            "Keep rotary_pos_emb input packing unchanged, but apply RoPE in the export adapter "
            "using Qwen3-style [B,H,S,D] / [B,1,S,D] broadcasting"
        ),
    )
    parser.add_argument(
        "cmd", choices=["prepare", "calibrate", "export-onnx", "export-fp16", "all"]
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.cmd == "prepare":
        prepare_config(args)
    elif args.cmd == "calibrate":
        calibrate_and_export_quant(args)
    elif args.cmd == "export-onnx":
        export_onnx(args)
    elif args.cmd == "export-fp16":
        export_raw_fp16_onnx(args)
    elif args.cmd == "all":
        run_all(args)
    else:
        raise ValueError(f"Unsupported cmd: {args.cmd}")


if __name__ == "__main__":
    main()
