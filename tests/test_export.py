from pkmn_rl_arena.export.passes.fusion_pass import (
    GemmQuantDequantFusionPass,
)
from pkmn_rl_arena.export.passes.delete_pass import (
    DeleteFirstInputQDQPass,
    DeleteQuantizePass,
    DeleteFirstLastQuantizeDequantizePass,
)
from pkmn_rl_arena.export.exporters.parameters import ExportParameters
from pkmn_rl_arena.export.onnx_exporter import ONNXExporter

from pkmn_rl_arena.quantize.quantize import FullQuantizer

from pkmn_rl_arena.data.parser import MapAnalyzer

from pkmn_rl_arena.paths import PATHS

import rustboyadvance_py

import sys
import os
import unittest

import numpy as np

import onnx
import onnxruntime as ort

import torch
import torch.nn as nn
from onnx import shape_inference
from onnx import numpy_helper
import torch.nn.functional as F


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)


def launch_makefile():
    makefile_path = os.path.join(
        os.path.dirname(__file__), "test_export_data/gba", "Makefile"
    )
    if not os.path.exists(makefile_path):
        raise FileNotFoundError(f"Makefile not found at {makefile_path}")

    os.system(f"make -C {os.path.dirname(makefile_path)}")


def setup_stop_addr(parser, gba):
    addr_write = int(parser.get_address("stopWriteData"), 16)
    addr_read = int(parser.get_address("stopReadData"), 16)
    gba.add_stop_addr(addr_write, 1, True, "stopWriteData", 3)
    gba.add_stop_addr(addr_read, 1, True, "stopReadData", 4)
    return addr_write, addr_read


def export_model_to_onnx(model, dummy_input, onnx_path, opset_version=11):
    """Export a PyTorch model to ONNX format."""
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )


def quantize_onnx_model(onnx_path, quantized_onnx_path, num_samples=10):
    """Quantize an ONNX model."""
    quantizer = FullQuantizer(onnx_path, quantized_onnx_path)
    calib_reader = FullQuantizer.create_fake_calibration_data(
        onnx_path, num_samples=num_samples
    )
    quantizer.quantize(calib_reader)

    # Infer shapes
    quantized_model = onnx.load(quantized_onnx_path)
    inferred_model = shape_inference.infer_shapes(quantized_model)
    onnx.save(inferred_model, quantized_onnx_path)


def apply_fusion_passes(
    model_path,
    output_path=None,
    use_gemm_fusion=True,
    use_delete_pass=True,
    use_delete_first_last_pass=False,
    use_delete_first_pass=True,
):
    """Apply fusion and optimization passes to an ONNX model."""
    if output_path is None:
        output_path = model_path

    onnx_model = onnx.load(model_path)

    if use_gemm_fusion:
        fusion_pass = GemmQuantDequantFusionPass()
        fusion_pass.run(onnx_model.graph)

    if use_delete_pass:
        delete_pass = DeleteQuantizePass()
        delete_pass.run(onnx_model.graph)

    if use_delete_first_last_pass:
        delete_first_pass = DeleteFirstLastQuantizeDequantizePass()
        delete_first_pass.run(onnx_model.graph)

    if use_delete_first_pass:
        delete_first_pass = DeleteFirstInputQDQPass()
        delete_first_pass.run(onnx_model.graph)

    onnx.save(onnx_model, output_path)
    return output_path


def setup_gba_environment(rom_path, map_path):
    """Setup GBA environment and return necessary objects."""
    gba = rustboyadvance_py.RustGba()
    gba.load(PATHS["BIOS"], rom_path)
    parser = MapAnalyzer(map_path)
    addr_write, addr_read = setup_stop_addr(parser, gba)

    output_addr = int(parser.get_address("output"), 16)
    input_addr = int(parser.get_address("input"), 16)

    return gba, parser, addr_write, addr_read, output_addr, input_addr


def run_gba_inference(
    gba, addr_write, input_addr, output_addr, input_data, output_size
):
    """Run inference on GBA and return results."""
    # Wait for initial stop
    id = gba.run_to_next_stop(20000)
    while id != 3:
        id = gba.run_to_next_stop(20000)

    # Write input data
    gba.write_i8_list(input_addr, input_data.reshape(-1).tolist())
    gba.write_u16(addr_write, 0)

    # Wait for computation to complete
    id = gba.run_to_next_stop(20000)
    while id != 4:
        id = gba.run_to_next_stop(20000)

    # Read output
    output_read = gba.read_i8_list(output_addr, output_size)
    return np.array(output_read, dtype=np.int8).reshape(-1)


def get_last_qdq_scaling_factor(graph):
    """
    Get the actual scale and zero_point value of the last DequantizeLinear node before output.
    Returns (scale_value, zero_point_value)
    """
    for node in reversed(graph.node):
        if node.op_type == "DequantizeLinear":
            # Find scale and zero_point names
            scale_name = node.input[1]
            zero_point_name = node.input[2]
            scale_value = None
            zero_point_value = None
            # Search initializers for actual values
            for init in graph.initializer:
                if init.name == scale_name:
                    scale_value = float(numpy_helper.to_array(init))
                if init.name == zero_point_name:
                    zero_point_value = int(numpy_helper.to_array(init))
            if scale_value is not None and zero_point_value is not None:
                return scale_value, zero_point_value
    return None, None


def get_first_qdq_scaling_factor(graph):
    """
    Get the actual scale and zero_point value of the first QuantizeLinear node after input.
    Returns (scale_value, zero_point_value)
    """
    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            # Find scale and zero_point names
            scale_name = node.input[1]
            zero_point_name = node.input[2]
            scale_value = None
            zero_point_value = None
            # Search initializers for actual values
            for init in graph.initializer:
                if init.name == scale_name:
                    scale_value = float(numpy_helper.to_array(init))
                if init.name == zero_point_name:
                    zero_point_value = int(numpy_helper.to_array(init))
            if scale_value is not None and zero_point_value is not None:
                return scale_value, zero_point_value
    return None, None


def run_onnx_inference(onnx_path, input_data):
    """Run inference on ONNX model and return int8 results."""
    ort_session = ort.InferenceSession(onnx_path)
    input_for_onnx = input_data.astype(np.float32)
    ort_inputs = {"input": input_for_onnx}
    ort_outs = ort_session.run(None, ort_inputs)
    return np.clip(np.round(ort_outs[0]), -128, 127).astype(np.int8).reshape(-1)


class TestExportParameters(unittest.TestCase):
    def setUp(self):
        self.template_path = os.path.join(
            os.path.dirname(__file__),
            "../pkmn_rl_arena/export/templates/parameters.jinja",
        )
        self.template_path = os.path.abspath(self.template_path)

    def test_export_array(self):
        array_data = [1, 2, 3, 4, 5]
        array_name = "test_array"
        output_path = "test_output.h"
        memory_region = "const"
        datatype = "int8_t"

        exporter = ExportParameters(
            template_path=self.template_path,
            array_data=np.array(array_data, dtype=np.int8),
            array_name=array_name,
            memory_region=memory_region,
            datatype=datatype,
            output_path=output_path,
        )
        exporter.export_array()

        with open(output_path, "r") as f:
            content = f.read()

        expected_content = (
            f"{memory_region} {datatype} {array_name}[] = {{1, 2, 3, 4, 5}};"
        )
        self.assertIn(expected_content, content)

        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)


class TestExportForward(unittest.TestCase):
    def setUp(self):
        self.template_path = os.path.join(
            os.path.dirname(__file__),
            "./../pkmn_rl_arena/export/templates/forward.jinja",
        )
        self.template_path = os.path.abspath(self.template_path)
        self.header_template_path = os.path.join(
            os.path.dirname(__file__),
            "./../pkmn_rl_arena/export/templates/forward_header.jinja",
        )
        self.header_template_path = os.path.abspath(self.header_template_path)
        self.template_parameters_path = os.path.join(
            os.path.dirname(__file__), "../templates/parameters.jinja"
        )
        self.template_parameters_path = os.path.abspath(self.template_parameters_path)
        self.rom_path = os.path.join(
            os.path.dirname(__file__), "./test_export_data/gba/gba.elf"
        )
        self.map_path = os.path.join(
            os.path.dirname(__file__), "./test_export_data/gba/build/gba.map"
        )
        
    def _test_model_inference(
        self,
        model,
        onnx_filename,
        input_shape,
        input_range,
        output_size,
        opset_version=11,
    ):
        """Helper method to test model inference comparison between ONNX and GBA."""
        dummy_input = torch.randn(*input_shape)
        onnx_path = onnx_filename

        export_model_to_onnx(model, dummy_input, onnx_path, opset_version)

        input_random = np.random.randint(*input_range, input_shape, dtype=np.int8)
        onnx_output = run_onnx_inference(onnx_path, input_random)

        exporter = ONNXExporter(onnx_path)
        exporter.export(
            output_dir=os.path.join(os.path.dirname(__file__), "test_export_data/gba")
        )
        launch_makefile()

        gba, parser, addr_write, addr_read, output_addr, input_addr = (
            setup_gba_environment(self.rom_path, self.map_path)
        )
        gba_output = run_gba_inference(
            gba, addr_write, input_addr, output_addr, input_random, output_size
        )

        print(f"ONNX output (int8): {onnx_output}")
        print(f"GBA output: {gba_output}")
        self.assertTrue(np.array_equal(onnx_output, gba_output))

    def test_export_forward_three_relu(self):
        class ThreeReLUOnly(nn.Module):
            def __init__(self):
                super(ThreeReLUOnly, self).__init__()
                self.relu1 = nn.ReLU()
                self.relu2 = nn.ReLU()
                self.relu3 = nn.ReLU()

            def forward(self, x):
                x = self.relu1(x)
                x = self.relu2(x)
                x = self.relu3(x)
                return x
        
        model = ThreeReLUOnly()
        self._test_model_inference(
            model=model,
            onnx_filename="three_relu_only.onnx",
            input_shape=(1, 10),
            input_range=(-128, 128),
            output_size=10
        )

    def test_export_fc_forward(self):
        class SingleFC(nn.Module):
            def __init__(self):
                super(SingleFC, self).__init__()
                self.fc = nn.Linear(10, 5, bias=True)
                self.fc.weight.data = torch.randint(-5, 10, self.fc.weight.shape).float()
                self.fc.bias.data = torch.randint(-5, 12, self.fc.bias.shape).float()

            def forward(self, x):
                return self.fc(x)

        model = SingleFC()
        self._test_model_inference(
            model=model,
            onnx_filename="single_fc.onnx",
            input_shape=(1, 10),
            input_range=(-18, 12),
            output_size=5
        )

    def test_qgemmcustom_op(self):
        class FCQuant(nn.Module):
            def __init__(self):
                super().__init__()
                self.w0 = nn.Parameter(torch.randn(10, 5))
                self.b0 = nn.Parameter(torch.randn(5))

            def forward(self, x):
                x = torch.matmul(x, self.w0) + self.b0
                return x

        model = FCQuant()
        dummy_input = torch.randn(1, 10)
        onnx_path = "fc.onnx"
        quantized_onnx_path = "fc_quant.onnx"
        fused_path = "fused_gemm_model.onnx"

        export_model_to_onnx(model, dummy_input, onnx_path, opset_version=13)
        quantize_onnx_model(onnx_path, quantized_onnx_path)

        quantized_model = onnx.load(quantized_onnx_path)

        output_scale = get_last_qdq_scaling_factor(quantized_model.graph)[0]

        if output_scale is None:
            raise ValueError("Could not find output DQ node scale and zero point")

        # Random input in float range between -1 and 1
        input_random = np.random.uniform(-1, 1, (1, 10)).astype(np.float32)

        ort_session = ort.InferenceSession(quantized_onnx_path)
        input_type = ort_session.get_inputs()[0].type

        if 'int8' in input_type:
            model_input = input_random
        else:
            model_input = input_random.astype(np.float32)

        ort_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: model_input})
        onnx_output = ort_outputs[0]

        apply_fusion_passes(quantized_onnx_path, fused_path,
                        use_gemm_fusion=True, use_delete_pass=True,
                        use_delete_first_pass=False, use_delete_first_last_pass=False)

        exporter = ONNXExporter(fused_path)
        exporter.export(output_dir=os.path.join(os.path.dirname(__file__), "test_export_data/gba"))
        launch_makefile()

        gba, parser, addr_write, addr_read, output_addr, input_addr = setup_gba_environment(
            self.rom_path, self.map_path)
        input_scale = get_first_qdq_scaling_factor(quantized_model.graph)[0]

        input_gba = np.round(input_random / input_scale).astype(np.int8)
        gba_output = run_gba_inference(gba, addr_write, input_addr, output_addr,
                                    input_gba, 5)  # 5 is output size

        print(f"ONNX output (int8): {onnx_output}")
        print(f"GBA output (int8): {gba_output}")

        onnx_float = onnx_output.astype(np.float32)
        gba_float = (gba_output.astype(np.float32) ) * output_scale

        print(f"ONNX output (float): {onnx_float}")
        print(f"GBA output (float): {gba_float}")

        float_match = np.allclose(onnx_float, gba_float, rtol=1e-2, atol=1e-2)

        if float_match:
            print("Dequantized outputs match within tolerance!")
        else:
            print(" Dequantized outputs don't match")

            # Print detailed differences
            diff = np.abs(onnx_float - gba_float)
            max_diff = np.max(diff)
            avg_diff = np.mean(diff)
            print(f"Max difference: {max_diff}")
            print(f"Average difference: {avg_diff}")
            print(f"Differences: {diff}")

        self.assertTrue(float_match, "Outputs don't match even after dequantization")


    def test_qgemm_relu_qgemm_relu(self):
        class TwoFCQuantRelu(nn.Module):
            def __init__(self):
                super().__init__()
                self.w0 = nn.Parameter(torch.randn(10, 8))
                self.b0 = nn.Parameter(torch.randn(8))
                self.w1 = nn.Parameter(torch.randn(8, 5))
                self.b1 = nn.Parameter(torch.randn(5))

            def forward(self, x):
                x = torch.matmul(x, self.w0) + self.b0
                x = F.relu(x)
                x = torch.matmul(x, self.w1) + self.b1
                x = F.relu(x)
                return x

        model = TwoFCQuantRelu()
        dummy_input = torch.randn(1, 10)
        onnx_path = "fc2.onnx"
        quantized_onnx_path = "fc2_quant.onnx"
        fused_path = "fused_gemm_relu_model.onnx"

        export_model_to_onnx(model, dummy_input, onnx_path, opset_version=13)
        quantize_onnx_model(onnx_path, quantized_onnx_path)
        quantized_model = onnx.load(quantized_onnx_path)
        output_scale = get_last_qdq_scaling_factor(quantized_model.graph)[0]
        if output_scale is None:
            raise ValueError("Could not find output DQ node scale and zero point")
        input_random = np.random.uniform(-1, 1, (1, 10)).astype(np.float32)
        ort_session = ort.InferenceSession(quantized_onnx_path)
        input_type = ort_session.get_inputs()[0].type
        if 'int8' in input_type:
            model_input = input_random
        else:
            model_input = input_random.astype(np.float32)
        ort_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: model_input})
        onnx_output = ort_outputs[0]
        apply_fusion_passes(quantized_onnx_path, fused_path, 
                            use_gemm_fusion=True, use_delete_pass=True, 
                            use_delete_first_pass=False, use_delete_first_last_pass=False)
        exporter = ONNXExporter(fused_path)
        exporter.export(output_dir=os.path.join(os.path.dirname(__file__), "test_export_data/gba"))
        launch_makefile()
        gba, parser, addr_write, addr_read, output_addr, input_addr = setup_gba_environment(
            self.rom_path, self.map_path)
        input_scale = get_first_qdq_scaling_factor(quantized_model.graph)[0]
        input_gba = np.round(input_random / input_scale).astype(np.int8)
        gba_output = run_gba_inference(gba, addr_write, input_addr, output_addr, 
                                    input_gba, 5)
        print(f"ONNX output (int8): {onnx_output}")
        print(f"GBA output (int8): {gba_output}")
        onnx_float = onnx_output.astype(np.float32)
        gba_float = (gba_output.astype(np.float32)) * output_scale
        print(f"ONNX output (float): {onnx_float}")
        print(f"GBA output (float): {gba_float}")
        float_match = np.allclose(onnx_float, gba_float, rtol=1e-2, atol=1e-2)
        if float_match:
            print("Dequantized outputs match within tolerance!")
        else:
            print("Dequantized outputs don't match")
            diff = np.abs(onnx_float - gba_float)
            max_diff = np.max(diff)
            avg_diff = np.mean(diff)
            print(f"Max difference: {max_diff}")
            print(f"Average difference: {avg_diff}")
            print(f"Differences: {diff}")
        self.assertTrue(float_match, "Outputs don't match even after dequantization")


    def test_qgemm_relu_qgemm_relu_large(self):
        class LargeFourFC(nn.Module):
            """
            4 fully-connected layers, built to be on the order of a few million parameters.
            Weight shapes use (in_features, out_features) to match the existing torch.matmul pattern.
            """
            def __init__(self):
                super().__init__()
                in0 = 1024
                h1 = 2048
                h2 = 1024
                h3 = 768
                h4 = 5
                self.w0 = nn.Parameter(torch.randn(in0, h1))
                self.b0 = nn.Parameter(torch.randn(h1))
                self.w1 = nn.Parameter(torch.randn(h1, h2))
                self.b1 = nn.Parameter(torch.randn(h2))
                self.w2 = nn.Parameter(torch.randn(h2, h3))
                self.b2 = nn.Parameter(torch.randn(h3))
                self.w3 = nn.Parameter(torch.randn(h3, h4))
                self.b3 = nn.Parameter(torch.randn(h4))

            def forward(self, x):
                x = torch.matmul(x, self.w0) + self.b0
                x = F.relu(x)
                x = torch.matmul(x, self.w1) + self.b1
                x = F.relu(x)
                x = torch.matmul(x, self.w2) + self.b2
                x = F.relu(x)
                x = torch.matmul(x, self.w3) + self.b3
                x = F.relu(x)
                return x

        model = LargeFourFC()
        print("Model total params:", sum(p.numel() for p in model.parameters()))
        dummy_input = torch.randn(1, 1024)
        onnx_path = "fc4_large.onnx"
        quantized_onnx_path = "fc4_large_quant.onnx"
        fused_path = "fused_gemm_relu_model.onnx"

        export_model_to_onnx(model, dummy_input, onnx_path, opset_version=13)
        quantize_onnx_model(onnx_path, quantized_onnx_path)
        quantized_model = onnx.load(quantized_onnx_path)
        output_scale = get_last_qdq_scaling_factor(quantized_model.graph)[0]
        if output_scale is None:
            raise ValueError("Could not find output DQ node scale and zero point")
        
        input_random = np.random.uniform(-1, 1, (1, 1024)).astype(np.float32)
        
        ort_session = ort.InferenceSession(quantized_onnx_path)
        input_type = ort_session.get_inputs()[0].type
        if "int8" in input_type:
            model_input = input_random
        else:
            model_input = input_random.astype(np.float32)
        ort_outputs = ort_session.run(
            None, {ort_session.get_inputs()[0].name: model_input}
        )
        onnx_output = ort_outputs[0]
        apply_fusion_passes(
            quantized_onnx_path,
            fused_path,
            use_gemm_fusion=True,
            use_delete_pass=True,
            use_delete_first_pass=False,
            use_delete_first_last_pass=False,
        )
        exporter = ONNXExporter(fused_path)
        exporter.export(
            output_dir=os.path.join(os.path.dirname(__file__), "test_export_data/gba")
        )
        launch_makefile()
        gba, parser, addr_write, addr_read, output_addr, input_addr = (
            setup_gba_environment(self.rom_path, self.map_path)
        )
        input_scale = get_first_qdq_scaling_factor(quantized_model.graph)[0]
        input_gba = np.round(input_random / input_scale).astype(np.int8)

        output_size = 5

        gba_output = run_gba_inference(gba, addr_write, input_addr, output_addr, 
                                    input_gba, output_size)
        print(f"ONNX output (int8): {onnx_output}")
        print(f"GBA output (int8): {gba_output}")
        onnx_float = onnx_output.astype(np.float32)
        gba_float = (gba_output.astype(np.float32)) * output_scale
        print(f"ONNX output (float): {onnx_float}")
        print(f"GBA output (float): {gba_float}")
        float_match = np.allclose(onnx_float, gba_float, rtol=1e-2, atol=1e-2)
        if float_match:
            print("Dequantized outputs match within tolerance!")
        else:
            print("Dequantized outputs don't match")
            diff = np.abs(onnx_float - gba_float)
            max_diff = np.max(diff)
            avg_diff = np.mean(diff)
            print(f"Max difference: {max_diff}")
            print(f"Average difference: {avg_diff}")
            print(f"Differences: {diff}")
        self.assertTrue(float_match, "Outputs don't match even after dequantization")


if __name__ == "__main__":
    unittest.main()
