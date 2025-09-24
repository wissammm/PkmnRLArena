from onnx import numpy_helper
import re

class OnnxUtils:
    @staticmethod
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

    @staticmethod
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
    
    @staticmethod
    def sanitize_name(name):
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    @staticmethod
    def sanitize_graph_names(graph):
        for node in graph.node:
            node.name = OnnxUtils.sanitize_name(node.name)
            node.op_type = OnnxUtils.sanitize_name(node.op_type)
            node.output[:] = [OnnxUtils.sanitize_name(out) for out in node.output]
            node.input[:] = [OnnxUtils.sanitize_name(inp) for inp in node.input]
        for tensor in graph.initializer:
            tensor.name = OnnxUtils.sanitize_name(tensor.name)
        for vi in graph.value_info:
            vi.name = OnnxUtils.sanitize_name(vi.name)
        for inp in graph.input:
            inp.name = OnnxUtils.sanitize_name(inp.name)
        for out in graph.output:
            out.name = OnnxUtils.sanitize_name(out.name)