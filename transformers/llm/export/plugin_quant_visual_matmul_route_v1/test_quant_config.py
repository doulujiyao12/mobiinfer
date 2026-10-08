import contextlib
import copy
import io
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import torch
from dopt.dopt_lm import do_opt

import visual_plugin_quant_matmul_route as route
from visual_plugin_quant_matmul_route import build_parser, patch_quant_config, quantization_report


def capture_quant_params(model):
    """Inspect official serialization using temporary files and scoped hooks."""
    with tempfile.TemporaryDirectory() as directory, \
            mock.patch.object(do_opt, "save_encrypt_file") as save_encrypt:
        do_opt.generate_quant_params(
            model, directory, quant_param_2=False, embedding_separate=False,
        )
        save_encrypt.assert_called_once()
        return copy.deepcopy(save_encrypt.call_args.args[0])


class QuantConfigTest(unittest.TestCase):
    def configure(self, directory, extra_args=(), w8=True):
        base_args = [
            "--quant_strategy", "Quant_aigc_ptq", "--weight_bit", "8",
            "--act_bit", "8", "--weight_algo", "min_max", "--input_algo", "min_max",
        ] if w8 else []
        args = build_parser().parse_args([*base_args, *extra_args, "prepare"])
        path = Path(directory) / "config.json"
        path.write_text(json.dumps({"layer_strategy": {"linear": {
            "type": "<class 'torch.nn.modules.linear.Linear'>", "quant_strategy": "float",
        }}}))
        patch_quant_config(
            str(path), args.quant_strategy, args.weight_bit, args.group_size,
            not args.omit_group_size, args.weight_algo, args.act_bit, args.input_algo,
            args.input_unsigned_quant, False, 16, True, "min_max",
        )
        model = torch.nn.Sequential(OrderedDict([("linear", torch.nn.Linear(128, 4))]))
        with contextlib.redirect_stdout(io.StringIO()):
            model = do_opt.optimize_model(model, str(path))
        return path, model

    def test_activation_choice_reaches_official_checkpoint_and_parameter_file(self):
        for flags, expected_unsigned, expected_dtype in [
            ((), True, "UINT8"),
            (("--input_unsigned_quant",), True, "UINT8"),
            (("--input_signed_quant",), False, "INT8"),
        ]:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                path, model = self.configure(directory, flags)
                self.assertNotIn("group_size", json.loads(path.read_text())["layer_strategy"]["linear"]["weight"])
                do_opt.set_quant_state(model, weight_state=True, input_state=True)
                do_opt.set_calibrate_state(model, True)
                with torch.no_grad():
                    model(torch.linspace(-3, 8, 384).reshape(3, 128))
                do_opt.set_calibrate_state(model, False)
                report = quantization_report(path, model.state_dict())
                self.assertEqual(report["input_unsigned_quant"], expected_unsigned)
                self.assertEqual((report["weight_bit"], report["act_bit"]), (8, 8))
                self.assertIsNone(report["group_size"])
                with contextlib.redirect_stdout(io.StringIO()):
                    plain = capture_quant_params(model)
                activation = plain["ModelLightWeightParameter"][0]["quant"]["input"][0]["QuantParam"][0]
                self.assertEqual(activation["dataType"], expected_dtype)
                if not expected_unsigned:
                    self.assertEqual(json.loads(activation["offset"]), [0.0])

    def test_report_detects_checkpoint_overriding_requested_signed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path, unsigned_model = self.configure(directory)
            old_state = unsigned_model.state_dict()
            path, signed_model = self.configure(directory, ("--input_signed_quant",))
            self.assertFalse(quantization_report(path)["input_unsigned_quant"])
            signed_model.load_state_dict(old_state, strict=True)
            self.assertTrue(quantization_report(path, signed_model.state_dict())["input_unsigned_quant"])

    def test_w4a16_keeps_grouped_signed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path, model = self.configure(directory, w8=False)
            report = quantization_report(path, model.state_dict())
            self.assertEqual((report["weight_bit"], report["act_bit"], report["group_size"]), (4, 16, 128))
            self.assertFalse(report["input_unsigned_quant"])

    def test_calibration_and_onnx_reports_follow_artifacts_instead_of_cli_defaults(self):
        class TinyVisual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 4)

            def forward(self, hidden_states_in, rotary_pos_emb, attention_mask):
                return self.linear(hidden_states_in)

        with tempfile.TemporaryDirectory() as directory:
            config, _model = self.configure(directory)
            config.rename(route.config_path(directory, 0))
            # Separately invoked commands can have CLI settings different from
            # the saved config. The reports must describe the actual artifacts.
            args = build_parser().parse_args([
                "--route_dir", directory, "--input_signed_quant", "calibrate",
            ])
            sample = {
                "hidden_states_in": torch.linspace(-3, 8, 384).reshape(1, 3, 128),
                "rotary_pos_emb": torch.zeros(2, 3, 1, 64),
                "attention_mask": torch.zeros(1, 3, 3),
            }
            meta = {"chunk_index": 0, "local_deepstack_count": 0}
            with mock.patch.object(route, "load_visual_chunk", side_effect=lambda *a, **kw: (TinyVisual(), meta)), \
                    mock.patch.object(route, "load_calibration_samples", return_value=([sample], [{"file": "test-input"}])), \
                    contextlib.redirect_stdout(io.StringIO()):
                route.calibrate_and_export_quant(args)
                route.export_onnx(args)
            for relative in ["quant_output/calibration_report.json", "onnx/export_report.json"]:
                report = json.loads((Path(directory) / relative).read_text())
                self.assertEqual((report["weight_bit"], report["act_bit"]), (8, 8))
                self.assertTrue(report["input_unsigned_quant"])
                self.assertIsNone(report["group_size"])
                self.assertEqual(report["quantization"]["source"], "calibrated_checkpoint")
                self.assertFalse(report["quant_param_2"])

    def test_native_w4_lut_is_paired_with_float32_onnx_and_rejects_stale_parameters(self):
        class TinyVisual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Keep a MatMul rather than allowing onnxsim to fold the tiny
                # biased layer into Gemm. Real ViT weights exceed the exporter's
                # 1 MB simplification threshold and retain named MatMul nodes.
                self.linear = torch.nn.Linear(128, 4, bias=False)

            def forward(self, hidden_states_in, rotary_pos_emb, attention_mask):
                return self.linear(hidden_states_in)

        with tempfile.TemporaryDirectory() as directory:
            config, _model = self.configure(directory, ("--group_size", "64"), w8=False)
            config.rename(route.config_path(directory, 0))
            args = build_parser().parse_args(["--route_dir", directory, "--quant_param_2", "calibrate"])
            sample = {"hidden_states_in": torch.linspace(-3, 8, 384).reshape(1, 3, 128),
                      "rotary_pos_emb": torch.zeros(2, 3, 1, 64), "attention_mask": torch.zeros(1, 3, 3)}
            meta = {"chunk_index": 0, "local_deepstack_count": 0}
            with mock.patch.object(route, "load_visual_chunk", side_effect=lambda *a, **kw: (TinyVisual(), meta)), \
                    mock.patch.object(route, "load_calibration_samples", return_value=([sample], [{"file": "test-input"}])), \
                    contextlib.redirect_stdout(io.StringIO()):
                route.calibrate_and_export_quant(args)
                calibrated = json.loads((Path(directory) / "quant_output/calibration_report.json").read_text())
                layer = calibrated["weight_encoding"]["layers"][0]
                self.assertEqual(layer["dtype"], "UINT4")
                self.assertTrue(layer["lut_mode"])
                self.assertTrue(layer["decoded_matches_reference"])
                weights = torch.load(route.fake_quant_weight_path(directory), weights_only=True)
                self.assertTrue(torch.equal(weights["linear.weight"], weights["linear.weight"].round()))
                # The production exporter replaces Linear with FLinearMatmul;
                # do the same in this tiny model rather than exporting torch's
                # fused bias form with a different node-name convention.
                def export_model(*positional, **keyword):
                    model = TinyVisual()
                    if keyword.get("prepare_export"):
                        route._replace_linear_with_flinear(model)
                    return model, meta
                with mock.patch.object(route, "load_visual_chunk", side_effect=export_model):
                    route.export_onnx(args)
                exported = json.loads((Path(directory) / "onnx/export_report.json").read_text())
                self.assertTrue(exported["paired_initializers_verified"])
                self.assertEqual(exported["quant_params_sha256"], calibrated["weight_encoding"]["quant_params_sha256"])
                args.fp16 = True
                with self.assertRaisesRegex(ValueError, "retain FLOAT32"):
                    route.export_onnx(args)
                args.fp16 = False
                with open(route.quant_params_path(directory), "ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(RuntimeError, "changed after calibration"):
                    route.export_onnx(args)

    def test_serialization_preserves_calibrated_model_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            _config, model = self.configure(directory, ("--group_size", "64"), w8=False)
            do_opt.set_quant_state(model, weight_state=True, input_state=True)
            do_opt.set_calibrate_state(model, True)
            with torch.no_grad():
                model(torch.randn(1, 3, 128))
            do_opt.set_calibrate_state(model, False)
            original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            route.ensure_route_layout(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                route.serialize_quantized_weights(model, directory, True)
            self.assertEqual(set(original), set(model.state_dict()))
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, original[name]), name)


if __name__ == "__main__":
    unittest.main()
