import re
from typing import Dict

class MapAnalyzer:
    def __init__(self, map_file):
        self.symbols : Dict[str , int]= {}
        self._parse(map_file)
    
    def _parse(self, map_file):
        pattern = re.compile(r'^\s*(0x[0-9a-fA-F]+)\s+(\S+)')
        
        with open(map_file, 'r') as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    addr, symbol = match.groups()
                    self.symbols[symbol] = addr
    
    def get_address(self, symbol) -> int | None :
        return self.symbols.get(symbol)
