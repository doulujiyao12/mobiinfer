import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx

from build_kirin_offline import Pipeline, deepstack_dup_candidate, parse_args, validate_calibration, validate_compile_log


class KirinOfflineTest(unittest.TestCase):
    def args(self, folder, *extra):
        return parse_args(["--model", str(folder / "hf"), "--ddk", str(folder / "sdk"),
                           "--output", str(folder / "runtime"), *extra])

    def test_json_configuration_is_overridden_by_cli_without_fixed_model_or_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "build.json"
            config.write_text(json.dumps({"model": str(root / "other-model"), "ddk": str(root / "sdk"),
                                          "output": str(root / "runtime"), "platform": "kirin9020",
                                          "npu_chunks": 4, "num_samples": 17}))
            args = parse_args(["--config", str(config), "--platform", "kirin9030", "--chunks", "1", "--steps", "compile"])
            self.assertEqual(args.platform, "kirin9030")
            self.assertEqual(args.model, root / "other-model")
            self.assertEqual(args.selected_chunks, [1])
            self.assertEqual((args.npu_chunks, args.num_samples), (4, 17))

    def test_json_rejects_unsupported_choices_and_boolean_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for invalid in ({"weight_bit": 3}, {"native_weight_encoding": "false"},
                            {"num_samples": 1.5}, {"act_bit": None}):
                config = root / "build.json"
                config.write_text(json.dumps({"model": str(root / "hf"), "ddk": str(root / "sdk"),
                                              "output": str(root / "runtime"), **invalid}))
                with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["--config", str(config)])

    def test_calibration_rejects_mixed_static_shapes_missing_samples_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            def sample(index, sequence=3, value=1.0):
                np.savez(folder / f"chunk_00_sample_{index:03d}.npz",
                         hidden_states_in=np.full((1, sequence, 8), value, dtype=np.float32),
                         rotary_pos_emb=np.zeros((2, sequence, 1, 4), dtype=np.float32),
                         attention_mask=np.zeros((1, sequence, sequence), dtype=np.float32))
            sample(0)
            with self.assertRaisesRegex(ValueError, "need 2"):
                validate_calibration(folder, [0], 2)
            sample(1, sequence=4)
            with self.assertRaisesRegex(ValueError, "shapes differ"):
                validate_calibration(folder, [0], 2)
            sample(1, value=np.nan)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_calibration(folder, [0], 2)
            sample(1)
            self.assertEqual(validate_calibration(folder, [0], 2)["hidden_states_in"], [1, 3, 8])

    def test_compiler_success_does_not_mask_cpu_fallback_or_fatal_diagnostics(self):
        success = "partition type NPU:1, CPU:0\nSaveCompiledModelToFile SUCCESS\nOMG generate offline model success"
        validate_compile_log(success)
        for text in (success.replace("CPU:0", "CPU:1"), success + "\nQuantizeOptimizer Fail!", "Done: model.omc"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                validate_compile_log(text)

    def test_deepstack_dup_is_derived_from_graph_instead_of_chunk_index_two(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hidden = onnx.helper.make_tensor_value_info("hidden_states", onnx.TensorProto.FLOAT, [1, 3, 8])
            deep = onnx.helper.make_tensor_value_info("deepstack_hidden_0", onnx.TensorProto.FLOAT, [1, 3, 8])
            graph = onnx.helper.make_graph([onnx.helper.make_node("Identity", ["hidden_states"], [deep.name])],
                                           "alias", [hidden], [hidden, deep])
            path = root / "chunk7.onnx"
            onnx.save(onnx.helper.make_model(graph), path)
            self.assertTrue(deepstack_dup_candidate(path))
            graph.node[0].op_type = "Neg"
            onnx.save(onnx.helper.make_model(graph), path)
            self.assertFalse(deepstack_dup_candidate(path))

    def test_compile_stages_paired_inputs_outside_sdk_rejected_dot_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = Pipeline(self.args(root, "--chunks", "0", "--steps", "compile"))
            (root / "sdk/tools/platform/kirin9030").mkdir(parents=True)
            route = pipeline.route(0)
            (route / "onnx").mkdir(parents=True)
            (route / "quant_output").mkdir()
            hidden = onnx.helper.make_tensor_value_info("hidden_states", onnx.TensorProto.FLOAT, [1, 3, 8])
            model = route / "onnx/model.onnx"
            onnx.save(onnx.helper.make_model(onnx.helper.make_graph([], "test", [hidden], [hidden])), model)
            (route / "onnx/weights.pb").write_bytes(b"paired external weights")
            (route / "quant_output/quant_params_file").write_bytes(b"paired parameters")
            (route / "route_config.json").write_text(json.dumps({"model": str(model), "input_shape": "test"}))
            report = {"onnx": str(model), "onnx_external_data_sha256": {"weights.pb": "test"},
                      "onnx_sha256": "model-hash", "quant_params_sha256": "parameter-hash"}
            def compiler(command, label, **kwargs):
                staging = Path(command[-2])
                self.assertNotIn(".", str(staging))
                self.assertNotIn("-", str(staging))
                config = json.loads((staging / "route_config.json").read_text())
                self.assertEqual(Path(config["model"]).read_bytes(), model.read_bytes())
                self.assertEqual((staging / "onnx/weights.pb").read_bytes(), b"paired external weights")
                self.assertEqual((staging / "quant_output/quant_params_file").read_bytes(), b"paired parameters")
                artifact = Path(kwargs["env"]["OUTPUT_PREFIX"] + ".omc")
                artifact.parent.mkdir()
                artifact.write_bytes(b"compiled graph")
                logs = pipeline.work / "logs"
                logs.mkdir()
                (logs / (label + ".log")).write_text("partition type NPU:1, CPU:0\nSaveCompiledModelToFile SUCCESS\nOMG generate offline model success")
            with mock.patch.object(pipeline, "paired_export", return_value=report), \
                    mock.patch.object(pipeline, "run", side_effect=compiler):
                pipeline.compile()
            built = json.loads((route / "compile_report.json").read_text())
            self.assertEqual(Path(built["artifact"]).read_bytes(), b"compiled graph")
            self.assertEqual(built["onnx_sha256"], report["onnx_sha256"])

    def test_output_cannot_overwrite_source_and_parameters_cannot_mix_between_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(root, "--output", str(root / "hf/runtime"))
            (root / "hf").mkdir()
            (root / "hf/config.json").write_text("{}")
            args = self.args(root, "--steps", "tools", "--mnnconvert", str(root / "MNNConvert"),
                             "--calib-input-dir", str(root / "inputs"))
            (root / "MNNConvert").touch()
            Pipeline(args).execute()
            changed = self.args(root, "--steps", "tools", "--mnnconvert", str(root / "MNNConvert"),
                                "--calib-input-dir", str(root / "inputs"), "--group-size", "128")
            with self.assertRaisesRegex(ValueError, "parameters changed"):
                Pipeline(changed).execute()


if __name__ == "__main__":
    unittest.main()
