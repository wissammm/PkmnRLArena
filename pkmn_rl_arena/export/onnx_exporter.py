from sympy import re
from .export_core.graph_loader import ONNXGraphLoader
from .export_core.memory_allocator import MemoryAllocator
from .export_core.exporter_factory import ExporterFactory
from .export_core.code_generator import CodeGenerator
from .passes.pass_manager import PassManager
from .passes.delete_pass import DeleteQuantizePass
from .onnx_utils import OnnxUtils

class ONNXExporter:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        
    def export(self, output_dir="gba"):
        # Load the model
        loader = ONNXGraphLoader(self.onnx_path)

        model = loader.load_model()
        OnnxUtils.sanitize_graph_names(model.graph)  

        value_info = loader.get_value_info()
        
        # Run optimization passes
        pass_manager = PassManager(model.graph)
        pass_manager.add_pass(DeleteQuantizePass())
        pass_manager.run_passes()


        optimized_graph = pass_manager.get_optimized_graph()
        
        updated_value_info = self._update_value_info(value_info, optimized_graph)
        
        allocator = MemoryAllocator(optimized_graph, updated_value_info)
        allocator.calculate_tensor_sizes()
        allocator.allocate_with_reuse()
        
        exporter_factory = ExporterFactory(
            optimized_graph, 
            value_info, 
            allocator.tensor_offsets
        )
        exporters = exporter_factory.create_exporters()
        
        code_generator = CodeGenerator(exporters, allocator)
        code_generator.generate(output_dir)

    
    
    def _update_value_info(self, value_info, graph):
        """
        Update value_info to match tensor names in the graph.
        Creates a mapping from quantized to regular tensor names.
        """
        updated_info = dict(value_info)

        
        # Find all tensor names in graph
        used_tensors = set()
        for node in graph.node:
            for inp in node.input:
                used_tensors.add(inp)
            for out in node.output:
                used_tensors.add(out)
        
        # For each tensor without shape info, try to find a matching shape
        for tensor in used_tensors:
            if tensor not in updated_info:
                # Look for base name in value_info (without _quantized suffix)
                if "_quantized" in tensor:
                    base_name = tensor.split("_quantized")[0]
                    if base_name in updated_info:
                        updated_info[tensor] = updated_info[base_name]
                # Look for tensors from same layer/operation
                parts = tensor.split('/')
                if len(parts) > 1:
                    for known_tensor in value_info:
                        if parts[1] in known_tensor:  # Match operation name
                            updated_info[tensor] = value_info[known_tensor]
                            break
        
        return updated_info