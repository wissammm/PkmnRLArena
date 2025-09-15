import pkmn_rl_arena.data.parser
import pkmn_rl_arena.data.pokemon_data
from pkmn_rl_arena.paths import PATHS
import rustboyadvance_py

import random
import unittest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)


MAIN_STEPS = 64000


class TestGbaFunctions(unittest.TestCase):
    def setUp(self):
        self.gba = rustboyadvance_py.RustGba()
        self.parser = pkmn_rl_arena.data.parser.MapAnalyzer(PATHS["MAP"])
        self.gba.load(PATHS["BIOS"], PATHS["ROM"])

    def test_read_u32(self):
        self.gba.add_stop_addr(
            int(self.parser.get_address("stopTestReadWrite"), 16),
            1,
            True,
            "stopTestReadWrite",
            12,
        )
        addr = int(self.parser.get_address("testBuffer"), 16)

        id = self.gba.run_to_next_stop(MAIN_STEPS)
        while id != 12:
            id = self.gba.run_to_next_stop(MAIN_STEPS)
        result = self.gba.read_u32(addr)
        # Check if the result equlas 3
        self.assertEqual(result, 3)

    def test_read_u32_list(self):
        self.gba.add_stop_addr(
            int(self.parser.get_address("stopTestReadWrite"), 16),
            1,
            True,
            "stopTestReadWrite",
            12,
        )
        addr = int(self.parser.get_address("listTestBuffer"), 16)
        id = self.gba.run_to_next_stop(MAIN_STEPS)
        while id != 12:
            id = self.gba.run_to_next_stop(MAIN_STEPS)

        result = self.gba.read_u32_list(addr, 6)
        expected = [10, 87, 76, 65, 1, 0]

        self.assertEqual(result, expected)

    def test_write_u32(self):
        self.gba.add_stop_addr(
            int(self.parser.get_address("stopTestReadWrite"), 16),
            1,
            True,
            "stopTestReadWrite",
            12,
        )
        addr = int(self.parser.get_address("testBuffer"), 16)
        id = self.gba.run_to_next_stop(MAIN_STEPS)
        while id != 12:
            id = self.gba.run_to_next_stop(MAIN_STEPS)
        self.gba.write_u32(addr, 42)
        result = self.gba.read_u32(addr)
        self.assertEqual(result, 42)

    def test_write_u32_list(self):
        self.gba.add_stop_addr(
            int(self.parser.get_address("stopTestReadWrite"), 16),
            1,
            True,
            "stopTestReadWrite",
            12,
        )
        addr = int(self.parser.get_address("listTestBuffer"), 16)
        id = self.gba.run_to_next_stop(MAIN_STEPS)
        while id != 12:
            id = self.gba.run_to_next_stop(MAIN_STEPS)
        data = [random.randint(0, 300) for _ in range(42)]

        self.gba.write_u32_list(addr, data)
        result = self.gba.read_u32_list(addr, len(data))
        self.assertEqual(result, data)

    def test_save_load_state(self):
        self.gba.add_stop_addr(
            int(self.parser.get_address("stopTestReadWriteTwo"), 16),
            1,
            True,
            "stopTestReadWriteTwo",
            9,
        )
        self.gba.add_stop_addr(
            int(self.parser.get_address("stopTestReadWrite"), 16),
            1,
            True,
            "stopTestReadWrite",
            12,
        )
        addr = int(self.parser.get_address("listTestBuffer"), 16)
        id = self.gba.run_to_next_stop(MAIN_STEPS)
        while id != 12:
            id = self.gba.run_to_next_stop(MAIN_STEPS)
        data = [random.randint(0, 300) for _ in range(42)]

        self.gba.write_u32_list(addr, data)

        # Save the state
        save_path = f"{PATHS["SAVE"]}/test_rustboy_advance_py.savestate"
        self.gba.save_savestate(save_path)


        # Load the state
        self.gba.load_savestate(save_path, PATHS["BIOS"], PATHS["ROM"])
        self.gba.add_stop_addr(
            int(self.parser.get_address("stopTestReadWriteTwo"), 16),
            1,
            True,
            "stopTestReadWriteTwo",
            9,
        )

        result = self.gba.read_u32_list(addr, len(data))
        self.assertEqual(result, data)
        self.gba.write_u16(
            int(self.parser.get_address("stopTestReadWrite"), 16), 0
        )  # Reset the stop condition

        id = self.gba.run_to_next_stop(MAIN_STEPS)
        max_try = 0
        while id == -1 and max_try < 10000:
            print(f"id  = {id} max_try = {max_try}")
            max_try = max_try + 1
            id = self.gba.run_to_next_stop(MAIN_STEPS)

        self.assertEqual(id, 9, "should be at the stop handle turn 2")


if __name__ == "__main__":
    unittest.main()
