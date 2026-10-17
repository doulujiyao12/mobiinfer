"""Keep imported GPTQ payloads from being bypassed by projection fusion."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmexport import LlmExporter, build_args
from utils.gptq import GPTQ, VisualGPTQ
from utils.transformers import Rotary


class GptqFusionTest(unittest.TestCase):
    def test_checkpoint_import_disables_fusion_before_model_construction(self):
        for option in ('--gptq_path', '--visual_gptq_path'):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                parser = argparse.ArgumentParser()
                build_args(parser)
                args = parser.parse_args(['--path', directory, '--dst_path', directory, option, directory])
                exporter = LlmExporter.__new__(LlmExporter)
                exporter.init_from_args(args)
                for name in ('transformer_c4', 'fuse_qkv_proj', 'fuse_gate_up_proj',
                             'fuse_ln_proj', 'fuse_linear_attn_gate'):
                    self.assertFalse(getattr(args, name), name)
                if option == '--gptq_path':
                    self.assertEqual(args.scale_bit, 32)

    def test_default_export_keeps_upstream_fusions(self):
        with tempfile.TemporaryDirectory() as directory:
            parser = argparse.ArgumentParser()
            build_args(parser)
            args = parser.parse_args(['--path', directory, '--dst_path', directory])
            exporter = LlmExporter.__new__(LlmExporter)
            exporter.init_from_args(args)
            self.assertTrue(args.transformer_c4)
            self.assertTrue(args.fuse_qkv_proj)

    def test_fused_graph_is_rejected_before_any_weight_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / 'graph.json'
            weights = Path(directory) / 'graph.weight'
            graph.write_text(json.dumps({'oplists': [{'type': 'FusedLinear', 'name': 'qkv'}]}))
            sentinel = b'checkpoint weights must remain unchanged'
            weights.write_bytes(sentinel)
            for importer in (GPTQ, VisualGPTQ):
                with self.subTest(importer=importer.__name__):
                    instance = importer.__new__(importer)
                    with self.assertRaisesRegex(ValueError, 'unfused projections'):
                        instance.apply(str(graph), str(weights))
                    self.assertEqual(weights.read_bytes(), sentinel)

    def test_fp16_scale_metadata_is_rejected_before_checkpoint_import(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / 'graph.json'
            weights = Path(directory) / 'graph.weight'
            graph.write_text(json.dumps({'oplists': [{
                'type': 'Convolution', 'name': '/layers.0/self_attn/q_proj/Linear',
                'main': {'external': [0, 16, 8, 0, 0],
                         'common': {'outputCount': 4, 'inputCount': 4},
                         'quanParameter': {'scaleStorage': 'FP16'}}}]}))
            weights.write_bytes(b'unchanged')
            importer = GPTQ.__new__(GPTQ)
            importer.weight_dict = {'0.self_attn.q_proj': object()}
            with self.assertRaisesRegex(ValueError, 'FP32 scale storage'):
                importer.apply(str(graph), str(weights))
            self.assertEqual(weights.read_bytes(), b'unchanged')

    def test_cpu_gptq_mrope_preserves_direct_fp32_angles(self):
        config = SimpleNamespace(rope_theta=1000000.0, rope_ratio=None, head_dim=128,
                                 model_type='qwen3_vl', max_position_embeddings=131072,
                                 rope_scaling={'mrope_section': [24, 20, 20], 'mrope_interleaved': True},
                                 export_args=SimpleNamespace(gptq_path='checkpoint'))
        rotary = Rotary(config)
        positions = torch.tensor([[0, 27, 2047, 4097], [0, 18, 2048, 4111], [0, 7, 2049, 4123]])
        theta = 1.0 / (1000000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
        axes = torch.zeros(64, dtype=torch.long)
        axes[1:60:3] = 1
        axes[2:60:3] = 2
        angles = (positions.float()[axes].T * theta).contiguous()
        expected = torch.stack([angles.cos(), angles.sin()])
        expected = torch.cat([expected, expected], dim=-1)
        actual = rotary(positions)[:, 0, :, 0, :]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
