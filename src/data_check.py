"""클래스 ID 매핑 확인과 이미지–라벨 정합성 검사.

원칙: 0단계에서는 '파이프라인 버그'(매핑 오류, 쌍 불일치, 포맷 오류)만 다룬다.
박스 경계·누락·오분류 같은 '라벨 품질' 문제는 1단계 몫이므로 여기서는
수정하지 않고 보고만 한다.
"""
from __future__ import annotations

import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# 좌표 정규화 오차 허용 범위 (라벨링 툴 반올림으로 1.0000001 등이 나올 수 있음)
COORD_TOL = 1e-3


# ---------------------------------------------------------------------------
# 클래스 이름 / ID 매핑
# ---------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """'Cool-Roof', 'cool_roof', 'cool roof' → 'coolroof' 로 정규화."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_class_names(path: str | Path) -> list[str]:
    """클래스 이름 목록을 읽는다.

    - Roboflow/Ultralytics ``data.yaml`` (names가 list 또는 {id: name} dict)
    - CVAT YOLO export의 ``obj.names`` (한 줄에 하나)
    """
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        with open(path, encoding="utf-8") as f:
            names = yaml.safe_load(f)["names"]
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=int)]
        return list(names)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def build_class_mapping(
    source_names: list[str],
    target_names: list[str] | dict[int, str],
    manual: dict[str, str] | None = None,
) -> dict[int, int]:
    """소스 데이터셋 클래스 ID → 목표 클래스 ID 매핑을 이름 기준으로 만든다.

    manual: 이름이 서로 다를 때 직접 지정 (예: {'white roof': 'cool roof'}).
    매칭되지 않는 소스 클래스가 있으면 조용히 넘어가지 않고 예외를 던진다.
    """
    if isinstance(target_names, dict):
        target_names = [target_names[k] for k in sorted(target_names)]
    target_idx = {normalize_name(n): i for i, n in enumerate(target_names)}
    manual_n = {normalize_name(k): normalize_name(v) for k, v in (manual or {}).items()}

    mapping, unmatched = {}, []
    for src_id, src_name in enumerate(source_names):
        key = manual_n.get(normalize_name(src_name), normalize_name(src_name))
        if key in target_idx:
            mapping[src_id] = target_idx[key]
        else:
            unmatched.append(f"{src_id}:{src_name!r}")
    if unmatched:
        raise ValueError(
            f"목표 클래스 {target_names}에 매칭되지 않는 소스 클래스: {unmatched} "
            f"— manual 인자로 직접 매핑을 지정하세요."
        )
    return mapping


def is_identity(mapping: dict[int, int]) -> bool:
    return all(k == v for k, v in mapping.items())


def describe_mapping(source_names, target_names, mapping) -> str:
    if isinstance(target_names, dict):
        target_names = [target_names[k] for k in sorted(target_names)]
    lines = []
    for s, t in sorted(mapping.items()):
        flag = "" if s == t else "   ⚠️ ID 변경 필요"
        lines.append(f"  {s} {source_names[s]!r:18} → {t} {target_names[t]!r}{flag}")
    head = "✅ 매핑 일치 (리매핑 불필요)" if is_identity(mapping) else "⚠️ 매핑 불일치 — 리매핑 필요"
    return head + "\n" + "\n".join(lines)


def remap_label_text(text: str, mapping: dict[int, int]) -> str:
    """라벨 파일 내용의 첫 컬럼(class id)만 매핑에 따라 바꾼다. 좌표는 그대로 둔다."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        if cls not in mapping:
            raise ValueError(f"매핑에 없는 class id {cls}: {line!r}")
        out.append(" ".join([str(mapping[cls])] + parts[1:]))
    return "\n".join(out) + ("\n" if out else "")


def copy_labels(src_dir: str | Path, dst_dir: str | Path,
                mapping: dict[int, int] | None = None) -> int:
    """라벨 폴더를 복사하면서 (지정 시) class id를 리매핑한다. 복사한 파일 수를 반환."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.glob("*.txt")):
        if mapping is None or is_identity(mapping):
            shutil.copy2(f, dst_dir / f.name)
        else:
            text = f.read_text(encoding="utf-8")
            (dst_dir / f.name).write_text(remap_label_text(text, mapping), encoding="utf-8")
        n += 1
    return n


# ---------------------------------------------------------------------------
# 이미지–라벨 정합성 검사
# ---------------------------------------------------------------------------
@dataclass
class SplitReport:
    split: str
    n_images: int = 0
    n_labels: int = 0
    images_without_label: list[str] = field(default_factory=list)
    labels_without_image: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)          # 배경 이미지일 수도 있음
    bad_lines: list[str] = field(default_factory=list)             # 파싱 불가·컬럼 수 오류
    out_of_range: list[str] = field(default_factory=list)          # 좌표가 0~1 밖
    unknown_class: list[str] = field(default_factory=list)
    duplicate_lines: list[str] = field(default_factory=list)       # 완전히 같은 줄 (1단계 참고용)
    format_counts: Counter = field(default_factory=Counter)        # 'bbox'(5컬럼) / 'polygon'(4점 OBB 등)
    class_counts: Counter = field(default_factory=Counter)

    @property
    def n_errors(self) -> int:
        """학습을 왜곡하는 '버그'성 문제 수 (빈 라벨·중복 줄은 경고로만 취급)."""
        return (len(self.images_without_label) + len(self.labels_without_image)
                + len(self.bad_lines) + len(self.out_of_range) + len(self.unknown_class))

    def summary(self, class_names: list[str] | None = None, max_items: int = 5) -> str:
        def fmt(items):
            more = f" … 외 {len(items) - max_items}건" if len(items) > max_items else ""
            return ", ".join(items[:max_items]) + more

        cls = {(class_names[k] if class_names and k < len(class_names) else k): v
               for k, v in sorted(self.class_counts.items())}
        status = "✅ 이상 없음" if self.n_errors == 0 else f"❌ 오류 {self.n_errors}건"
        lines = [
            f"[{self.split}] {status} | 이미지 {self.n_images} / 라벨 {self.n_labels}",
            f"  포맷: {dict(self.format_counts)} | 클래스 분포: {cls}",
        ]
        for name, items in [
            ("라벨 없는 이미지", self.images_without_label),
            ("이미지 없는 라벨", self.labels_without_image),
            ("파싱/컬럼 오류", self.bad_lines),
            ("좌표 범위 초과", self.out_of_range),
            ("미정의 클래스", self.unknown_class),
            ("(경고) 빈 라벨", self.empty_labels),
            ("(경고·1단계 참고) 중복 줄", self.duplicate_lines),
        ]:
            if items:
                lines.append(f"  - {name} {len(items)}건: {fmt(items)}")
        return "\n".join(lines)


def _parse_line(parts: list[str]):
    """(class_id, coords, kind) 반환. kind: 'bbox' | 'polygon'. 형식 오류 시 ValueError."""
    if len(parts) < 5:
        raise ValueError("컬럼 수 부족")
    cls = float(parts[0])
    if cls != int(cls):
        raise ValueError("class id가 정수가 아님")
    coords = [float(x) for x in parts[1:]]
    if len(coords) == 4:
        kind = "bbox"
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        kind = "polygon"   # OBB(4점=8값) 또는 세그먼트 폴리곤
    else:
        raise ValueError(f"좌표 개수 {len(coords)}개 — bbox(4) 또는 짝수 폴리곤이어야 함")
    return int(cls), coords, kind


def check_split(split_dir: str | Path, num_classes: int, split: str | None = None) -> SplitReport:
    """``split_dir/images`` 와 ``split_dir/labels`` 의 정합성을 검사한다."""
    split_dir = Path(split_dir)
    rep = SplitReport(split=split or split_dir.name)
    images = {p.stem: p for p in (split_dir / "images").glob("*") if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in (split_dir / "labels").glob("*.txt")}
    rep.n_images, rep.n_labels = len(images), len(labels)
    rep.images_without_label = sorted(set(images) - set(labels))
    rep.labels_without_image = sorted(set(labels) - set(images))

    for stem, lp in sorted(labels.items()):
        raw = [ln.strip() for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not raw:
            rep.empty_labels.append(stem)
            continue
        for ln, dup in Counter(raw).items():
            if dup > 1:
                rep.duplicate_lines.append(f"{stem} (x{dup})")
        for i, line in enumerate(raw, 1):
            try:
                cls, coords, kind = _parse_line(line.split())
            except ValueError as e:
                rep.bad_lines.append(f"{stem}:{i} {e}")
                continue
            rep.format_counts[kind] += 1
            rep.class_counts[cls] += 1
            if not 0 <= cls < num_classes:
                rep.unknown_class.append(f"{stem}:{i} cls={cls}")
            if any(c < -COORD_TOL or c > 1 + COORD_TOL for c in coords):
                rep.out_of_range.append(f"{stem}:{i}")
            elif kind == "bbox" and (coords[2] <= 0 or coords[3] <= 0):
                rep.out_of_range.append(f"{stem}:{i} (w/h ≤ 0)")
    return rep


def check_dataset(datasets_dir: str | Path, class_names: list[str],
                  splits=("train", "val", "test"), verbose: bool = True) -> dict[str, SplitReport]:
    """여러 스플릿을 한 번에 검사하고 요약을 출력한다. 라벨 폴더가 없는 스플릿은 건너뛴다."""
    datasets_dir = Path(datasets_dir)
    reports = {}
    for s in splits:
        if not (datasets_dir / s / "images").exists():
            continue
        if not (datasets_dir / s / "labels").exists() or not any((datasets_dir / s / "labels").glob("*.txt")):
            if verbose:
                n = sum(1 for p in (datasets_dir / s / "images").glob("*") if p.suffix.lower() in IMAGE_EXTS)
                print(f"[{s}] 라벨 없음 — 이미지 {n}장 (추론 전용 스플릿)")
            continue
        reports[s] = check_split(datasets_dir / s, len(class_names), s)
        if verbose:
            print(reports[s].summary(class_names))
    return reports
