"""지각 해시(dHash) 기반 스플릿 간 근접 중복 이미지 탐지.

같은 지역 타일이나 같은 원본에서 나온 이미지가 train과 val에 함께 있으면
검증 성능이 실제보다 높게 나온다(데이터 누수). 기본 제공 200장과 Roboflow
139장의 출처가 겹칠 수 있으므로, B0를 측정하기 전에 한 번 점검한다.

항공·위성 영상은 회전·반전 증강본이 흔하므로 8가지 2면체 변환(90° 회전 × 반전)
중 최소 해밍 거리로 비교한다. 외부 라이브러리 없이 Pillow + NumPy만 사용한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .data_check import IMAGE_EXTS


def _dhash_bits(gray: np.ndarray) -> np.ndarray:
    """(8, 9) 회색조 배열 → 64비트 bool 배열 (인접 픽셀 밝기 차이 부호)."""
    return (gray[:, 1:] > gray[:, :-1]).flatten()


def dihedral_hashes(path: str | Path, size: int = 8) -> np.ndarray:
    """이미지 하나의 8가지 회전·반전 변환별 dHash. shape = (8, size*size) bool."""
    with Image.open(path) as im:
        base = im.convert("L")
        # 정사각형으로 먼저 줄여야 90° 회전 후에도 같은 격자를 비교할 수 있다.
        base = base.resize((size + 1, size + 1), Image.Resampling.LANCZOS)
    arr = np.asarray(base, dtype=np.float32)
    variants = []
    for flip in (False, True):
        a = np.fliplr(arr) if flip else arr
        for k in range(4):
            r = np.rot90(a, k)
            variants.append(_dhash_bits(r[:size, :]))
    return np.stack(variants)


@dataclass
class DuplicatePair:
    split_a: str
    image_a: str
    split_b: str
    image_b: str
    distance: int    # 해밍 거리 (0 = 사실상 동일)


def _list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMAGE_EXTS)


def find_cross_split_duplicates(
    split_dirs: dict[str, str | Path],
    pairs: list[tuple[str, str]] | None = None,
    threshold: int = 6,
) -> list[DuplicatePair]:
    """스플릿 간 근접 중복 이미지 쌍을 찾는다.

    split_dirs: {'train': .../train/images, 'val': .../val/images, ...}
    pairs     : 비교할 스플릿 쌍. 기본은 모든 조합.
    threshold : 64비트 중 다른 비트 수 허용치. 0~5 거의 동일, 6~10 의심 수준.
    """
    names = list(split_dirs)
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    files = {s: _list_images(Path(d)) for s, d in split_dirs.items()}
    hashes = {}
    for s, fs in files.items():
        if fs:
            # 원본 쪽은 변환 0(그대로)만, 비교 대상 쪽은 8가지 변환을 모두 사용
            hashes[s] = np.stack([dihedral_hashes(f) for f in fs])  # (N, 8, 64)

    found = []
    for a, b in pairs:
        if a not in hashes or b not in hashes:
            continue
        ha = hashes[a][:, 0, :]                      # (Na, 64)
        hb = hashes[b]                               # (Nb, 8, 64)
        # (Na, Nb, 8) 해밍 거리 → 변환 축 최소값
        dist = (ha[:, None, None, :] != hb[None, :, :, :]).sum(-1).min(-1)
        for i, j in zip(*np.nonzero(dist <= threshold)):
            found.append(DuplicatePair(a, files[a][i].name, b, files[b][j].name, int(dist[i, j])))
    return sorted(found, key=lambda p: (p.distance, p.split_a, p.image_a))


def report_duplicates(dups: list[DuplicatePair], max_items: int = 20) -> str:
    if not dups:
        return "✅ 스플릿 간 근접 중복 이미지 없음"
    lines = [f"⚠️ 스플릿 간 근접 중복 의심 {len(dups)}쌍 (거리 작을수록 동일):"]
    for p in dups[:max_items]:
        lines.append(f"  d={p.distance:2d}  {p.split_a}/{p.image_a}  ↔  {p.split_b}/{p.image_b}")
    if len(dups) > max_items:
        lines.append(f"  … 외 {len(dups) - max_items}쌍")
    lines.append("  → 육안 확인 후 실제 중복이면 B0 측정 전에 한쪽 스플릿으로 몰아 주세요.")
    return "\n".join(lines)
