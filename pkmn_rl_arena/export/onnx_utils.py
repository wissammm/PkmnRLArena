from onnx import numpy_helper

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