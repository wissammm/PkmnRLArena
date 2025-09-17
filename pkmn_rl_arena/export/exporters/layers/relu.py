from typing import Dict, List

from pkmn_rl_arena.export.base import LayerExporter
from ...enums import CallPosition

class ReLUExporter(LayerExporter):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, input_idx: int=0, output_idx: int=0, datatype : str = "int8_t", call_position: CallPosition = CallPosition.BETWEEN):
        self.name = name.replace("/", "_").upper()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.datatype = datatype
        self.call_position = call_position
        
    def get_function_call(self) -> str:
        if self.call_position == CallPosition.FIRST:
            return f"relu_{self.datatype}(input, mem_buffer + {self.name.upper()}_OUTPUT_IDX,{self.name.upper()}_SIZE);"
        elif self.call_position == CallPosition.LAST:
            return f"relu_{self.datatype}(mem_buffer + {self.name.upper()}_INPUT_IDX, output,{self.name.upper()}_SIZE);"
        elif self.call_position == CallPosition.BETWEEN:
            return f"relu_{self.datatype}(mem_buffer + {self.name.upper()}_INPUT_IDX, mem_buffer + {self.name.upper()}_OUTPUT_IDX,{self.name.upper()}_SIZE);"
        elif self.call_position == CallPosition.BOTH:
            return f"relu_{self.datatype}(input, output);"
        else:
            raise ValueError("Invalid call position specified.")
    
    def export_layer(self, output_dir: str) -> Dict[str, str]:
        """Export ReLU layer implementation."""
        files_generated = {}

    def get_defines(self) -> List[str]:
        """Get the list of defines for the ReLU layer."""
        return [
            f" {self.name.upper()}_INPUT_IDX {self.input_idx}",
            f" {self.name.upper()}_OUTPUT_IDX {self.output_idx}",
            f" {self.name.upper()}_SIZE {self.input_shape[1]}",
        ]
    
    def get_include(self) -> str:
        """Get the include statement for the ReLU layer."""
        return f""
    
