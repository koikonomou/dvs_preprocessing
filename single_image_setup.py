#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, argparse
from typing import Tuple, List, Optional
import pygame
from PIL import Image

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, argparse
from typing import Tuple, List, Optional
import pygame
from PIL import Image

def init_display(fullscreen: bool, display_index: Optional[int], window_size: str):
    pygame.display.init()
    pygame.font.init()
    if fullscreen and display_index is not None:
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(display_index)

    if fullscreen:
        info = pygame.display.Info()
        size = (info.current_w, info.current_h)
        flags = pygame.FULLSCREEN
    else:
        try:
            w, h = map(int, window_size.lower().split("x"))
        except Exception:
            w, h = 1024, 768
        size = (w, h)
        flags = 0

    screen = pygame.display.set_mode(size, flags)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    return screen, clock, size, font

def load_image_surface(path: str, fit: str):
    """
    fit:
      - 'resize'    -> stretch to 512x512
      - 'letterbox' -> preserve aspect inside 512x512 with black padding
      - 'none'      -> original size, no scaling/padding
    Returns (surface, mapping dict).
    """
    im = Image.open(path).convert("RGB")
    iw, ih = im.size

    if fit == "resize":
        target = 512
        im2 = im.resize((target, target), Image.BICUBIC)
        surf = pygame.image.frombuffer(im2.tobytes(), (target, target), "RGB")
        mapping = dict(mode="resize", in_w=iw, in_h=ih, out_w=target, out_h=target,
                       scale_x=target/iw, scale_y=target/ih, pad_x=0, pad_y=0)

    elif fit == "letterbox":
        target = 512
        scale = min(target / iw, target / ih)
        nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        im2 = im.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (target, target), (0, 0, 0))
        pad_x = (target - nw) // 2
        pad_y = (target - nh) // 2
        canvas.paste(im2, (pad_x, pad_y))
        surf = pygame.image.frombuffer(canvas.tobytes(), (target, target), "RGB")
        mapping = dict(mode="letterbox", in_w=iw, in_h=ih, out_w=target, out_h=target,
                       scale_x=scale, scale_y=scale, pad_x=pad_x, pad_y=pad_y)

    else:  # 'none'
        surf = pygame.image.frombuffer(im.tobytes(), (iw, ih), "RGB")
        mapping = dict(mode="none", in_w=iw, in_h=ih, out_w=iw, out_h=ih,
                       scale_x=1.0, scale_y=1.0, pad_x=0, pad_y=0)

    return surf, mapping

def rcls_positions(img_size: Tuple[int,int], disp_size: Tuple[int,int],
                   loops: int, step_px: int, speed_px_s: int, hz: int) -> List[Tuple[int,int]]:
    iw, ih = img_size; dw, dh = disp_size
    cx = (dw - iw) // 2
    cy = (dh - ih) // 2
    rel = [(0,0), (step_px,-step_px), (2*step_px,0), (step_px,step_px), (0,0)]
    seq = []
    dt = 1.0 / float(hz)
    vx = float(speed_px_s); vy = float(speed_px_s)
    for _ in range(loops):
        for (x0, y0), (x1, y1) in zip(rel[:-1], rel[1:]):
            dx, dy = (x1 - x0), (y1 - y0)
            seg_t = max(abs(dx)/vx if vx>0 else 0.05, abs(dy)/vy if vy>0 else 0.05)
            steps = max(1, int(round(seg_t / dt)))
            for s in range(1, steps+1):
                x = x0 + (s/steps) * dx
                y = y0 + (s/steps) * dy
                seq.append((cx + int(round(x)), cy + int(round(y))))
    return seq

def main():
    ap = argparse.ArgumentParser(description="Display one image with RCLS motion. No camera used.")
    ap.add_argument("image", help="Path to a single image (jpg/png/...)")
    ap.add_argument("--fit", choices=["resize","letterbox","none"], default="resize",
                    help="resize=stretch to 512, letterbox=preserve aspect in 512, none=original size")
    ap.add_argument("--hz", type=int, default=60)
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--step", type=int, default=12)
    ap.add_argument("--speed", type=int, default=180)
    ap.add_argument("--settle", type=float, default=0.4)
    ap.add_argument("--transition", type=float, default=2.0)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--display-index", type=int, default=None)
    ap.add_argument("--window-size", type=str, default="1024x768")
    args = ap.parse_args()

    screen, clock, disp_size, font = init_display(
        fullscreen=not args.windowed,
        display_index=args.display_index,
        window_size=args.window_size,
    )

    surf, mapping = load_image_surface(args.image, fit=args.fit)
    iw, ih = surf.get_size()
    cx = (disp_size[0] - iw) // 2
    cy = (disp_size[1] - ih) // 2

    # settle
    screen.fill((0,0,0)); screen.blit(surf, (cx, cy))
    label = font.render(f"{args.fit} settle...", True, (220, 220, 220))
    screen.blit(label, (20, 20)); pygame.display.flip()
    t0 = time.time()
    while time.time() - t0 < args.settle:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.display.quit(); return
        clock.tick(args.hz)

    # motion
    positions = rcls_positions((iw, ih), disp_size, args.loops, args.step, args.speed, args.hz)
    for (ox, oy) in positions:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.display.quit(); return
        screen.fill((0,0,0)); screen.blit(surf, (ox, oy))
        pygame.display.flip(); clock.tick(args.hz)

    # transition
    screen.fill((0,0,0))
    msg = font.render("done. (ESC to exit)", True, (180, 220, 180))
    screen.blit(msg, (20, 20)); pygame.display.flip()
    t1 = time.time()
    while time.time() - t1 < args.transition:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.display.quit(); return
        clock.tick(args.hz)

    pygame.display.quit()

if __name__ == "__main__":
    main()
