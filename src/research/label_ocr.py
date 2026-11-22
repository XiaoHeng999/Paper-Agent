"""Bounded local text hints for horizontal and vertical plot labels."""
import io
import shutil
import subprocess
import time

from PIL import Image, ImageOps


def label_views(png, deadline):
    executable = shutil.which("tesseract")
    if not executable:
        return {}
    with Image.open(io.BytesIO(png)) as image:
        scale = min(2, 2400/max(image.size))
        gray = ImageOps.grayscale(image).resize((max(1, round(image.width*scale)),
            max(1, round(image.height*scale))), Image.Resampling.BICUBIC)
    readings = {}
    for angle in (0, 90, 270):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        output = io.BytesIO()
        gray.rotate(angle, expand=True).save(output, format="PNG")
        try:
            run = subprocess.run([executable, "stdin", "stdout", "--psm", "11"], input=output.getvalue(),
                capture_output=True, check=True, timeout=min(4, remaining))
        except (OSError, subprocess.SubprocessError):
            continue
        text = run.stdout.decode("utf-8", errors="replace").strip()
        if text:
            readings[str(angle)] = text[:2400]
    return readings
