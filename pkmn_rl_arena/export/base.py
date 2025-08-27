import os
import jinja2
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod

class Exporter:
    """Base class for all exporters that use Jinja templates."""
    
    def __init__(self, template_path: str):
        """
        Initialize with the path to a Jinja template.
        
        Args:
            template_path: Path to the Jinja template file
        """
        self.template_path = template_path
        self.template_dir = os.path.dirname(template_path)
        self.template_name = os.path.basename(template_path)
        
        # Set up Jinja environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        self.template = self.env.get_template(self.template_name)


class LayerExporter(ABC):
    """Abstract base class for layer exporters."""
    
    @abstractmethod
    def export_layer(self, output_dir: str) -> Dict[str, str]:
        """
        Export layer and return generated files info.
        
        Args:
            output_dir: Directory where files should be saved
            
        Returns:
            Dictionary of generated files with their paths
        """
        pass
    
    @abstractmethod
    def get_function_call(self) -> str:
        """
        Get the C function call for this layer.
        
        Returns:
            String containing the C function call
        """
        pass
    
    @abstractmethod
    def get_defines(self) -> List[str]:
        """
        Get the list of defines for the layer.
        
        Returns:
            List of define statements
        """
        pass
    
    @abstractmethod
    def get_include(self) -> List[str]:
        """
        Get the list of header filenames required for the layer (no '#include').
        
        Returns:
            List of header filenames as strings
        """
        pass