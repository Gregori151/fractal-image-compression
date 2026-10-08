import math
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image


MAGIC = b"FRA2"
FORMAT_VERSION = 3

HEADER_STRUCT = struct.Struct("<4sBBIIHHHHffI")

RECORD_STRUCT_GRAY = struct.Struct("<HHHHHBxff")
RECORD_STRUCT_RGB = struct.Struct("<HHHHHBx3f3f")

TRANSFORM_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_identity",
    "flip_rot90",
    "flip_rot180",
    "flip_rot270",
)


class FractalRecord(object):
    def __init__(self, x, y, size, domain_x, domain_y, transform_id,
                 contrast, brightness, error=0.0):
        self.x = int(x)
        self.y = int(y)
        self.size = int(size)
        self.domain_x = int(domain_x)
        self.domain_y = int(domain_y)
        self.transform_id = int(transform_id)
        self.error = float(error)

        if np.isscalar(contrast):
            self.contrast = float(contrast)
            self.brightness = float(brightness)
        else:
            self.contrast = np.asarray(contrast, dtype=np.float64).reshape(3)
            self.brightness = np.asarray(brightness, dtype=np.float64).reshape(3)


class FractalCode(object):
    def __init__(self, width, height, channels, max_block_size,
                 min_block_size, threshold, max_abs_contrast,
                 iterations, domain_stride, records):
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.max_block_size = int(max_block_size)
        self.min_block_size = int(min_block_size)
        self.threshold = float(threshold)
        self.max_abs_contrast = float(max_abs_contrast)
        self.iterations = int(iterations)
        self.domain_stride = 0 if domain_stride is None else int(domain_stride)
        self.records = records


class CompressionMetrics(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def load_image(path):
    image = Image.open(str(path))
    if image.mode in ("1", "L", "I", "I;16", "F"):
        image = image.convert("L")
    else:
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def save_image(array, path):
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:
        image = Image.fromarray(array, mode="RGB")
    else:
        raise ValueError("Unsupported image array shape: {}".format(array.shape))
    image.save(str(path))


def image_channels(image):
    return 1 if image.ndim == 2 else image.shape[2]


def validate_config(image, max_block_size, min_block_size, domain_stride,
                     threshold, max_abs_contrast):
    if image.ndim not in (2, 3):
        raise ValueError("Only grayscale or RGB images are supported.")
    if image.dtype != np.uint8:
        raise ValueError("Image array must have dtype uint8.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Color images must have exactly 3 channels.")
    if max_block_size <= 0 or min_block_size <= 0:
        raise ValueError("Block sizes must be positive.")
    if min_block_size > max_block_size:
        raise ValueError("min_block_size must be <= block_size.")
    if max_block_size % min_block_size != 0:
        raise ValueError("block_size must be divisible by min_block_size.")

    b = max_block_size
    while b > 1:
        if b % 2 != 0:
            raise ValueError("block_size must be a power of 2.")
        b //= 2
    b = min_block_size
    while b > 1:
        if b % 2 != 0:
            raise ValueError("min_block_size must be a power of 2.")
        b //= 2

    height, width = image.shape[:2]
    if width % max_block_size != 0 or height % max_block_size != 0:
        raise ValueError(
            "Image width and height must be divisible by block_size."
        )
    if width < 2 * min_block_size or height < 2 * min_block_size:
        raise ValueError("Image is too small for domain blocks.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    if not (0 < max_abs_contrast < 1):
        raise ValueError("max_abs_contrast must be in (0, 1).")
    if domain_stride is not None and domain_stride <= 0:
        raise ValueError("domain_stride must be positive.")


def apply_transform(block, transform_id):
    if transform_id < 0 or transform_id >= 8:
        raise ValueError("transform_id must be in [0, 7].")
    if transform_id < 4:
        return np.rot90(block, k=transform_id, axes=(0, 1))
    return np.flip(np.rot90(block, k=transform_id - 4, axes=(0, 1)), axis=1)


def downsample_domain(domain):
    h, w = domain.shape[:2]
    if h % 2 or w % 2:
        raise ValueError("Domain dimensions must be even.")
    if domain.ndim == 2:
        return domain.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    c = domain.shape[2]
    return domain.reshape(h // 2, 2, w // 2, 2, c).mean(axis=(1, 3))


def make_domain_positions(image_shape, block_size, stride=None):
    height, width = image_shape[:2]
    domain_size = 2 * block_size
    if stride is None:
        stride = block_size
    if width < domain_size or height < domain_size:
        return []
    positions = []
    for y in range(0, height - domain_size + 1, stride):
        for x in range(0, width - domain_size + 1, stride):
            positions.append((x, y))
    return positions


def prepare_level_candidates(image, block_size, domain_stride):
    positions = make_domain_positions(image.shape, block_size, domain_stride)
    if not positions:
        raise ValueError(
            "No domain blocks available for block size {}.".format(block_size)
        )

    transformed = []
    for x, y in positions:
        ds = downsample_domain(
            image[y:y + 2 * block_size, x:x + 2 * block_size]
        ).astype(np.float32)
        for transform_id in range(8):
            transformed.append(
                apply_transform(ds, transform_id).astype(np.float32)
            )

    # Shape: [num_domains * 8, pixels, channels]
    if image.ndim == 2:
        arr = np.stack(transformed, axis=0)
        arr = arr.reshape(arr.shape[0], block_size * block_size, 1)
    else:
        arr = np.stack(transformed, axis=0)
        arr = arr.reshape(arr.shape[0], block_size * block_size, 3)

    means = arr.mean(axis=1)
    centered = arr - means[:, None, :]
    variances = np.mean(centered * centered, axis=1)
    return positions, arr, means, variances


def find_best_match(range_block, level_data, max_abs_contrast,
                    threshold, chunk_size=512):
    positions, candidates, means, variances = level_data
    r = range_block.astype(np.float32, copy=False)
    if r.ndim == 2:
        r = r.reshape(r.shape[0] * r.shape[1], 1)
    else:
        r = r.reshape(r.shape[0] * r.shape[1], 3)

    mean_r = r.mean(axis=0)
    centered_r = r - mean_r
    var_r = np.mean(centered_r * centered_r, axis=0)

    total = candidates.shape[0]
    best_error = math.inf
    best_index = -1
    best_contrast = None
    best_brightness = None
    comparisons = 0

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        d = candidates[start:end]
        md = means[start:end]
        vd = variances[start:end]
        cd = d - md[:, None, :]

        covariance = np.mean(cd * centered_r[None, :, :], axis=1)
        contrast = np.zeros_like(covariance, dtype=np.float32)
        valid = vd > 1e-12
        contrast[valid] = covariance[valid] / vd[valid]
        contrast = np.clip(contrast, -max_abs_contrast, max_abs_contrast)
        brightness = mean_r[None, :] - contrast * md

        mse_per_channel = (
            var_r[None, :]
            + contrast * contrast * vd
            - 2.0 * contrast * covariance
        )
        mse = mse_per_channel.mean(axis=1)

        local = int(np.argmin(mse))
        local_error = float(mse[local])
        comparisons += (end - start)

        if local_error < best_error:
            best_error = local_error
            best_index = start + local
            best_contrast = contrast[local].astype(np.float64)
            best_brightness = brightness[local].astype(np.float64)

        if best_error <= threshold:
            break

    if best_index < 0:
        raise RuntimeError("Failed to find a domain match.")

    if range_block.ndim == 2:
        best_contrast = float(best_contrast[0])
        best_brightness = float(best_brightness[0])

    domain_index = best_index // 8
    transform_id = best_index % 8
    return (
        positions[domain_index],
        transform_id,
        best_contrast,
        best_brightness,
        best_error,
        comparisons,
    )


def encode(image, block_size=32, min_block_size=8,
           threshold=100.0, max_abs_contrast=0.99,
           domain_stride=None, verbose=False):
    validate_config(
        image, block_size, min_block_size, domain_stride,
        threshold, max_abs_contrast
    )

    start = time.time()
    cache = {}
    records = []
    comparisons = 0
    level_counts = {}

    def get_level(size):
        if size not in cache:
            stride = size if domain_stride is None else domain_stride
            cache[size] = prepare_level_candidates(image, size, stride)
        return cache[size]

    initial = []
    height, width = image.shape[:2]
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            initial.append((x, y, block_size))

    queue = initial
    processed = 0
    accepted = 0
    split_count = 0

    while queue:
        x, y, size = queue.pop()
        range_block = image[y:y + size, x:x + size]
        level_data = get_level(size)
        (
            (domain_x, domain_y),
            transform_id,
            contrast,
            brightness,
            best_error,
            used_comparisons,
        ) = find_best_match(
            range_block,
            level_data,
            max_abs_contrast,
            threshold,
        )
        comparisons += used_comparisons
        processed += 1

        should_split = (
            best_error > threshold and size > min_block_size
        )

        if should_split:
            half = size // 2
            queue.append((x, y, half))
            queue.append((x + half, y, half))
            queue.append((x, y + half, half))
            queue.append((x + half, y + half, half))
            split_count += 1
        else:
            records.append(
                FractalRecord(
                    x, y, size, domain_x, domain_y,
                    transform_id, contrast, brightness, best_error
                )
            )
            accepted += 1
            level_counts[size] = level_counts.get(size, 0) + 1

        if verbose and processed % 50 == 0:
            print(
                "Processed nodes: {} | leaf blocks: {} | queued: {}".format(
                    processed, accepted, len(queue)
                )
            )

    elapsed = time.time() - start
    channels = image_channels(image)
    code = FractalCode(
        width,
        height,
        channels,
        block_size,
        min_block_size,
        threshold,
        max_abs_contrast,
        8,
        domain_stride,
        records,
    )

    domain_blocks_total = sum(
        len(cache[size][0]) for size in cache
    )

    stats = {
        "encode_seconds": elapsed,
        "range_blocks": len(records),
        "domain_blocks": domain_blocks_total,
        "comparisons": comparisons,
        "split_count": split_count,
        "processed_nodes": processed,
        "leaf_counts": level_counts,
        "levels_used": sorted(cache.keys(), reverse=True),
    }
    return code, stats


def decode(code, iterations=None, initial_value=128.0):
    if iterations is None:
        iterations = code.iterations
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if code.channels not in (1, 3):
        raise ValueError("Unsupported number of channels.")

    height, width = code.height, code.width
    shape = (height, width) if code.channels == 1 else (height, width, 3)
    current = np.full(shape, initial_value, dtype=np.float64)
    start = time.time()

    for _ in range(iterations):
        next_image = np.empty_like(current)
        for record in code.records:
            size = record.size
            domain_size = 2 * size
            domain = current[
                record.domain_y:record.domain_y + domain_size,
                record.domain_x:record.domain_x + domain_size,
            ]
            if domain.shape[0] != domain_size or domain.shape[1] != domain_size:
                raise ValueError("Invalid domain coordinates in fractal code.")

            reduced = downsample_domain(domain)
            transformed = apply_transform(reduced, record.transform_id)

            if code.channels == 1:
                restored = record.contrast * transformed + record.brightness
            else:
                restored = (
                    transformed * record.contrast.reshape(1, 1, 3)
                    + record.brightness.reshape(1, 1, 3)
                )

            next_image[
                record.y:record.y + size,
                record.x:record.x + size,
            ] = restored

        current = np.clip(next_image, 0.0, 255.0)

    elapsed = time.time() - start
    return np.rint(current).astype(np.uint8), elapsed


def save_code(code, path):
    path = Path(path)
    if code.channels not in (1, 3):
        raise ValueError("Only grayscale and RGB are supported.")

    if code.domain_stride < 0 or code.domain_stride > 65535:
        raise ValueError("domain_stride out of uint16 range.")

    with path.open("wb") as f:
        header = HEADER_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION,
            code.channels,
            code.width,
            code.height,
            code.max_block_size,
            code.min_block_size,
            code.domain_stride,
            code.iterations,
            float(code.threshold),
            float(code.max_abs_contrast),
            len(code.records),
        )
        f.write(header)

        for record in code.records:
            if code.channels == 1:
                f.write(
                    RECORD_STRUCT_GRAY.pack(
                        record.x, record.y, record.size,
                        record.domain_x, record.domain_y,
                        record.transform_id,
                        float(record.contrast),
                        float(record.brightness),
                    )
                )
            else:
                c = np.asarray(record.contrast, dtype=np.float64).reshape(3)
                b = np.asarray(record.brightness, dtype=np.float64).reshape(3)
                f.write(
                    RECORD_STRUCT_RGB.pack(
                        record.x, record.y, record.size,
                        record.domain_x, record.domain_y,
                        record.transform_id,
                        float(c[0]), float(c[1]), float(c[2]),
                        float(b[0]), float(b[1]), float(b[2]),
                    )
                )

    return path.stat().st_size


def load_code(path):
    path = Path(path)
    with path.open("rb") as f:
        data = f.read(HEADER_STRUCT.size)
        if len(data) != HEADER_STRUCT.size:
            raise ValueError("Invalid fractal code header.")

        (
            magic, version, channels, width, height,
            max_block_size, min_block_size, domain_stride,
            iterations, threshold, max_abs_contrast, record_count,
        ) = HEADER_STRUCT.unpack(data)

        if magic != MAGIC or version != FORMAT_VERSION:
            raise ValueError("Unsupported or invalid fractal code format.")
        if channels not in (1, 3):
            raise ValueError("Unsupported channel count.")

        records = []
        record_struct = (
            RECORD_STRUCT_GRAY if channels == 1 else RECORD_STRUCT_RGB
        )

        for _ in range(record_count):
            raw = f.read(record_struct.size)
            if len(raw) != record_struct.size:
                raise ValueError("Unexpected end of fractal code.")

            if channels == 1:
                (
                    x, y, size, dx, dy, transform_id,
                    contrast, brightness,
                ) = RECORD_STRUCT_GRAY.unpack(raw)
                records.append(
                    FractalRecord(
                        x, y, size, dx, dy, transform_id,
                        contrast, brightness
                    )
                )
            else:
                (
                    x, y, size, dx, dy, transform_id,
                    cr, cg, cb, br, bg, bb,
                ) = RECORD_STRUCT_RGB.unpack(raw)
                records.append(
                    FractalRecord(
                        x, y, size, dx, dy, transform_id,
                        np.array([cr, cg, cb], dtype=np.float64),
                        np.array([br, bg, bb], dtype=np.float64),
                    )
                )

    return FractalCode(
        int(width), int(height), int(channels),
        int(max_block_size), int(min_block_size),
        float(threshold), float(max_abs_contrast), int(iterations),
        None if domain_stride == 0 else int(domain_stride), records
    )


def calculate_mse(original, restored):
    if original.shape != restored.shape:
        raise ValueError("Images must have identical shapes.")
    a = original.astype(np.float64)
    b = restored.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def calculate_psnr(mse, max_value=255.0):
    if mse == 0:
        return math.inf
    return float(10.0 * math.log10((max_value * max_value) / mse))


def compress_image(input_path, code_path, restored_path,
                   block_size=32, min_block_size=8,
                   threshold=100.0, max_abs_contrast=0.99,
                   domain_stride=None, iterations=8, verbose=False):
    original = load_image(input_path)
    code, enc_stats = encode(
        original,
        block_size=block_size,
        min_block_size=min_block_size,
        threshold=threshold,
        max_abs_contrast=max_abs_contrast,
        domain_stride=domain_stride,
        verbose=verbose,
    )
    code.iterations = iterations

    code_bytes = save_code(code, code_path)
    loaded = load_code(code_path)
    restored, decode_seconds = decode(loaded, iterations=iterations)
    save_image(restored, restored_path)

    mse = calculate_mse(original, restored)
    psnr = calculate_psnr(mse)
    raw_bytes = int(original.nbytes)
    source_bytes = int(Path(input_path).stat().st_size)

    ratio_raw = float(raw_bytes) / float(code_bytes)
    ratio_file = float(source_bytes) / float(code_bytes)
    reduction = (1.0 - float(code_bytes) / float(raw_bytes)) * 100.0

    return CompressionMetrics(
        source_width=int(original.shape[1]),
        source_height=int(original.shape[0]),
        channels=int(code.channels),
        max_block_size=int(block_size),
        min_block_size=int(min_block_size),
        range_blocks=int(enc_stats["range_blocks"]),
        domain_blocks=int(enc_stats["domain_blocks"]),
        comparisons=int(enc_stats["comparisons"]),
        split_count=int(enc_stats["split_count"]),
        processed_nodes=int(enc_stats["processed_nodes"]),
        leaf_counts=enc_stats["leaf_counts"],
        levels_used=enc_stats["levels_used"],
        encode_seconds=float(enc_stats["encode_seconds"]),
        decode_seconds=float(decode_seconds),
        raw_pixel_bytes=raw_bytes,
        source_file_bytes=source_bytes,
        code_bytes=int(code_bytes),
        compression_ratio_raw=float(ratio_raw),
        compression_ratio_file=float(ratio_file),
        compression_percentage_raw=float(reduction),
        mse=mse,
        psnr_db=psnr,
    )


def print_metrics(m):
    print("\n=== Adaptive fractal compression metrics ===")
    print("Image size:              {} x {}".format(m.source_width, m.source_height))
    print("Channels:                {} ({})".format(
        m.channels, "grayscale" if m.channels == 1 else "RGB"
    ))
    print("Max range block:         {} x {}".format(
        m.max_block_size, m.max_block_size
    ))
    print("Min range block:         {} x {}".format(
        m.min_block_size, m.min_block_size
    ))
    print("Leaf range blocks:       {}".format(m.range_blocks))
    print("Processed nodes:         {}".format(m.processed_nodes))
    print("Split operations:        {}".format(m.split_count))
    print("Leaf blocks by size:      {}".format(m.leaf_counts))
    print("Domain blocks prepared:  {}".format(m.domain_blocks))
    print("Domain comparisons:      {}".format(m.comparisons))
    print("Raw pixel data:          {} bytes".format(m.raw_pixel_bytes))
    print("Source file:             {} bytes".format(m.source_file_bytes))
    print("Fractal code:            {} bytes".format(m.code_bytes))
    print("Compression ratio (raw): {:.4f}".format(m.compression_ratio_raw))
    print("Compression ratio (file):{:.4f}".format(m.compression_ratio_file))
    print("Data reduction:          {:.2f}%".format(m.compression_percentage_raw))
    print("MSE:                     {:.4f}".format(m.mse))
    if math.isinf(m.psnr_db):
        print("PSNR:                    inf dB")
    else:
        print("PSNR:                    {:.4f} dB".format(m.psnr_db))
    print("Encoding time:           {:.4f} s".format(m.encode_seconds))
    print("Decoding time:           {:.4f} s".format(m.decode_seconds))
    print("Levels used:             {}".format(m.levels_used))


# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================
# block_size — максимальный размер range-блока.
# min_block_size — минимальный размер; ниже блок не делится.
# Если ошибка > threshold, блок делится на 4 части.

CONFIG = {
    "input_path": "volsu.bmp",   # Сжимаемое изображение
    "code_path": "adaptive_result.fci",
    "restored_path": "adaptive_restored.bmp",

    "block_size": 32,
    "min_block_size": 4,
    "threshold": 100.0,
    "max_abs_contrast": 0.99,
    "domain_stride": None,
    "iterations": 8,
    "verbose": True,
}


def main():
    input_path = Path(CONFIG["input_path"])
    code_path = Path(CONFIG["code_path"])
    restored_path = Path(CONFIG["restored_path"])

    code_path.parent.mkdir(parents=True, exist_ok=True)
    restored_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            "Исходное изображение не найдено: {}".format(input_path.resolve())
        )

    print("=" * 60)
    print("АДАПТИВНОЕ ФРАКТАЛЬНОЕ СЖАТИЕ ИЗОБРАЖЕНИЯ")
    print("RGB + grayscale")
    print("=" * 60)
    print("Исходное изображение: {}".format(input_path.resolve()))
    print("Максимальный block_size: {}".format(CONFIG["block_size"]))
    print("Минимальный block_size:  {}".format(CONFIG["min_block_size"]))
    print("threshold:               {}".format(CONFIG["threshold"]))
    print("domain_stride:           {}".format(CONFIG["domain_stride"]))
    print("iterations:              {}".format(CONFIG["iterations"]))

    metrics = compress_image(
        input_path=input_path,
        code_path=code_path,
        restored_path=restored_path,
        block_size=CONFIG["block_size"],
        min_block_size=CONFIG["min_block_size"],
        threshold=CONFIG["threshold"],
        max_abs_contrast=CONFIG["max_abs_contrast"],
        domain_stride=CONFIG["domain_stride"],
        iterations=CONFIG["iterations"],
        verbose=CONFIG["verbose"],
    )
    print_metrics(metrics)
    print("\nКод:      {}".format(code_path.resolve()))
    print("Результат: {}".format(restored_path.resolve()))


if __name__ == "__main__":
    main()
