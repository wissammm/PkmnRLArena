import arcade
import numpy as np
import rustboyadvance_py
from PIL import Image

ROM_PATH = "/home/wboussella/Documents/rl_new_pokemon_ai/rl_new_pokemon_ai/pokeemerald_ai_rl/pokeemerald_modern.elf"
BIOS_PATH = "/home/wboussella/Documents/rl_new_pokemon_ai/rl_new_pokemon_ai/rustboyadvance-ng-for-rl/gba_bios.bin"
STEPS = 32000

import pygame
import numpy as np

class GBADisplay:
    def __init__(self, width=240, height=160, scale=3):
        pygame.init()
        self.width = width
        self.height = height
        self.scale = scale
        self.screen = pygame.display.set_mode((width * scale, height * scale))
        pygame.display.set_caption("GBA Display")

    def render(self, buffer):
        # reshape height width 
        array = np.array(buffer, dtype=np.uint32).reshape(self.height, self.width)

        bgra = array.view(np.uint8).reshape(self.height, self.width, 4)
        # bgra to rgb for pygame
        rgb = bgra[..., [2, 1, 0]]  # R, G, B

        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))  # (width, height, 3)
        scaled = pygame.transform.scale(surface, 
                                        (self.width * self.scale, 
                                         self.height * self.scale))
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()



def main():
    
    display = GBADisplay() 
    gba = rustboyadvance_py.RustGba()
    gba.load(BIOS_PATH, ROM_PATH)
    while True:

        steps_does = gba.run_to_next_stop(STEPS)

        buffer = gba.get_frame_buffer() 
        display.render(buffer)
        gba.run_to_next_stop(STEPS) 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

if __name__ == "__main__":
    main()