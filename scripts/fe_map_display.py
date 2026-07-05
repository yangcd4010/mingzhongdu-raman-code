from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def validate_crop(crop: tuple[int, int, int, int] | None, width: int, height: int) -> tuple[int, int, int, int]:
    if crop is None:
        return (0, 0, width, height)
    left, top, right, bottom = crop
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"Crop {crop} is outside image bounds width={width}, height={height}.")
    return crop


def effective_intensity(image: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    if image.mode in {"RGB", "RGBA"}:
        rgba = np.asarray(image.convert("RGBA"))
        rgb = rgba[..., :3].astype(np.float32)
        intensity = rgb.max(axis=2)
        return intensity, rgb
    return np.asarray(image.convert("F"), dtype=np.float32), None


def stretch_to_uint8(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    scaled = (values.astype(np.float32) - lower) / (upper - lower)
    return np.rint(np.clip(scaled, 0, 1) * 255.0).astype(np.uint8)


def make_red_display(crop_image: Image.Image, lower_percentile: float, upper_percentile: float) -> Image.Image:
    intensity, rgb = effective_intensity(crop_image)
    finite = intensity[np.isfinite(intensity)]
    if finite.size == 0:
        raise ValueError("No finite pixels found in selected crop.")
    lower = float(np.percentile(finite, lower_percentile))
    upper = float(np.percentile(finite, upper_percentile))
    if upper <= lower:
        lower, upper = float(finite.min()), float(finite.max())
    stretched = stretch_to_uint8(intensity, lower, upper)

    if rgb is None:
        out = np.zeros((*stretched.shape, 3), dtype=np.uint8)
        out[..., 0] = stretched
        return Image.fromarray(out, mode="RGB")

    denom = np.maximum(intensity, 1e-6)
    factor = stretched.astype(np.float32) / denom
    adjusted = np.clip(rgb * factor[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted, mode="RGB")


def parse_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = [int(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Crop must be left,top,right,bottom.")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display-only linear stretch for an SEM-EDS Fe map image.")
    parser.add_argument("input_image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/fe_map_display.png"))
    parser.add_argument("--crop", type=parse_crop, default=None, help="Optional left,top,right,bottom crop.")
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    args = parser.parse_args()

    image = Image.open(args.input_image)
    crop = validate_crop(args.crop, image.width, image.height)
    crop_image = image.crop(crop)
    display = make_red_display(crop_image, args.lower_percentile, args.upper_percentile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    display.save(args.output)
    print(f"Saved display image to: {args.output}")


if __name__ == "__main__":
    main()

