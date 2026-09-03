#!/usr/bin/env python3
"""OpenCV preprocessing for the original-color Mendeley PCB images.

The default mode is a small smoke run.  Each image gets all intermediate
stages and overlapping 640x640 final tiles; use --limit 0 for the full set.
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np


def crop_board(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = ((hsv[:, :, 1] > 20) | (gray < 245)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w * h < img.shape[0] * img.shape[1] * 0.15:
        return img
    pad = max(5, int(0.02 * max(w, h)))
    x0, y0 = max(0, x-pad), max(0, y-pad)
    x1, y1 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
    return img[y0:y1, x0:x1]


def binary_lines(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    t, _ = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Dark structures become black, background white.  Border statistics fix
    # the occasional inverted Otsu result.
    out = np.where(clahe < t, 0, 255).astype(np.uint8)
    border = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]])
    if float((border == 0).mean()) > 0.5:
        out = 255 - out
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return clahe, out


def save_tiles(img, directory, stem, size, overlap):
    directory.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    stride = max(1, size - overlap)
    xs = list(range(0, max(1, w-size+1), stride))
    ys = list(range(0, max(1, h-size+1), stride))
    if not xs or xs[-1] != max(0, w-size): xs.append(max(0, w-size))
    if not ys or ys[-1] != max(0, h-size): ys.append(max(0, h-size))
    count = 0
    for y in ys:
        for x in xs:
            tile = img[y:min(y+size,h), x:min(x+size,w)]
            canvas = np.full((size, size), 255, np.uint8)
            canvas[:tile.shape[0], :tile.shape[1]] = tile
            cv2.imwrite(str(directory / f"{stem}_x{x}_y{y}.png"), canvas)
            count += 1
    return count


def process(path, out_root, tile_size, overlap, resize_size=None, do_crop=False):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"cannot read {path}")
    d = out_root / path.stem
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d/'00_original.jpg'), img)
    # Default keeps the complete original image.  Cropping is opt-in because
    # automatic foreground crops can remove useful edge defects.
    crop = crop_board(img) if do_crop else img
    if resize_size:
        crop = cv2.resize(crop, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(d/'01_crop.jpg'), crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY); cv2.imwrite(str(d/'02_gray.png'), gray)
    denoise = cv2.GaussianBlur(gray, (3,3), 0); cv2.imwrite(str(d/'03_denoise.png'), denoise)
    contrast, final = binary_lines(denoise)
    cv2.imwrite(str(d/'04_contrast.png'), contrast)
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(d/'05_threshold.png'), thresh)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    cv2.imwrite(str(d/'06_morphology.png'), morph)
    cv2.imwrite(str(d/'07_final.png'), final)
    n = save_tiles(final, d/'tiles', path.stem, tile_size, overlap)
    return {'image': str(path), 'output': str(d), 'original_size': list(img.shape[:2][::-1]), 'processed_size': list(crop.shape[:2][::-1]), 'tiles': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, default=Path('data/raw/mendeley_pcb_defect/PCB-Defect An Annotated Dataset for Surface Defect/extracted/PCB_Defect/images'))
    ap.add_argument('--output-dir', type=Path, default=Path('outputs/preprocess_smoke'))
    ap.add_argument('--limit', type=int, default=3, help='0 processes all images')
    ap.add_argument('--tile-size', type=int, default=640); ap.add_argument('--overlap', type=int, default=64)
    ap.add_argument('--resize-size', type=int, default=None, help='resize complete image to NxN, e.g. 640; no random crop')
    ap.add_argument('--crop-board', action='store_true', help='opt in to largest-foreground crop')
    args = ap.parse_args(); files = sorted(args.input_dir.glob('*.jpg'))
    if args.limit > 0: files = files[:args.limit]
    rows = [process(p, args.output_dir, args.tile_size, args.overlap, args.resize_size, args.crop_board) for p in files]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'preprocess_report.json').write_text(json.dumps({'count':len(rows), 'tile_size':args.tile_size, 'overlap':args.overlap, 'resize_size':args.resize_size, 'crop_board':args.crop_board, 'items':rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'count':len(rows), 'output_dir':str(args.output_dir), 'items':rows}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
