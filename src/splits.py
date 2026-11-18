"""Validation 이미지 목록 고정과 라벨 버전 관리.

- 이미지 목록(val_list.txt)은 한 번 정하면 바꾸지 않는다 → 모든 실험의 비교 기준.
- 라벨은 1단계에서 수정될 수 있으므로 내용 해시로 버전을 붙인다.
  val_v1(원본), val_v2(검수본) … 처럼 이름과 해시를 레지스트리(JSON)에 기록하고,
  한 번 등록된 버전 이름은 다른 내용으로 덮어쓸 수 없다(불변).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .data_check import IMAGE_EXTS


def list_images(images_dir: str | Path) -> list[str]:
    return sorted(p.name for p in Path(images_dir).glob("*") if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# 이미지 목록 고정
# ---------------------------------------------------------------------------
def save_image_list(images_dir: str | Path, out_path: str | Path, overwrite: bool = False) -> list[str]:
    """현재 이미지 파일명 목록을 저장한다. 이미 있으면 덮어쓰지 않고 검증만 한다."""
    out_path = Path(out_path)
    names = list_images(images_dir)
    if out_path.exists() and not overwrite:
        verify_image_list(images_dir, out_path)
        print(f"✅ 고정 목록 {out_path.name} 과 일치 ({len(names)}장) — 기존 목록 유지")
        return names
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"📌 이미지 목록 고정: {out_path} ({len(names)}장)")
    return names


def verify_image_list(images_dir: str | Path, list_path: str | Path) -> None:
    """현재 폴더의 이미지 구성이 고정 목록과 정확히 같은지 확인. 다르면 AssertionError."""
    expected = set(Path(list_path).read_text(encoding="utf-8").split())
    actual = set(list_images(images_dir))
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    assert not missing and not extra, (
        f"검증셋 구성이 고정 목록과 다릅니다 — 누락 {len(missing)}장 {missing[:5]}, "
        f"추가 {len(extra)}장 {extra[:5]}"
    )


# ---------------------------------------------------------------------------
# 라벨 버전
# ---------------------------------------------------------------------------
def label_fingerprint(labels_dir: str | Path, image_names: list[str] | None = None) -> dict:
    """라벨 내용의 해시와 기본 통계. 줄 순서·공백 차이는 무시한다(내용이 같으면 같은 해시)."""
    labels_dir = Path(labels_dir)
    stems = ([Path(n).stem for n in image_names] if image_names is not None
             else sorted(p.stem for p in labels_dir.glob("*.txt")))
    h = hashlib.sha256()
    n_objects = 0
    for stem in sorted(stems):
        f = labels_dir / f"{stem}.txt"
        lines = []
        if f.exists():
            lines = sorted(" ".join(ln.split()) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
        n_objects += len(lines)
        h.update(stem.encode())
        h.update(b"\0")
        h.update("\n".join(lines).encode())
        h.update(b"\1")
    return {"hash": h.hexdigest()[:16], "n_images": len(stems), "n_objects": n_objects}


def _load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def register_label_version(labels_dir: str | Path, registry_path: str | Path, name: str,
                           image_names: list[str] | None = None, note: str = "") -> dict:
    """라벨 버전을 등록한다. 같은 이름이 다른 해시로 이미 있으면 예외(버전 불변 원칙)."""
    registry_path = Path(registry_path)
    reg = _load_registry(registry_path)
    fp = label_fingerprint(labels_dir, image_names)
    if name in reg:
        if reg[name]["hash"] != fp["hash"]:
            raise ValueError(
                f"{name}은 이미 다른 내용(hash {reg[name]['hash']})으로 등록되어 있습니다. "
                f"라벨을 수정했다면 새 버전 이름(예: val_v{len(reg) + 1})으로 등록하세요."
            )
        print(f"✅ {name} 확인 — hash {fp['hash']} ({fp['n_images']}장 / 객체 {fp['n_objects']}개)")
        return reg[name]
    for other, info in reg.items():
        if info["hash"] == fp["hash"]:
            print(f"ℹ️ 내용이 기존 버전 {other}와 동일합니다.")
    reg[name] = {**fp, "created": datetime.now().isoformat(timespec="seconds"), "note": note}
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📌 라벨 버전 등록: {name} — hash {fp['hash']} ({fp['n_images']}장 / 객체 {fp['n_objects']}개)")
    return reg[name]


def identify_label_version(labels_dir: str | Path, registry_path: str | Path,
                           image_names: list[str] | None = None) -> str:
    """현재 라벨이 레지스트리의 어떤 버전인지 찾는다. 미등록 상태면 예외 — 기록 누락 방지."""
    fp = label_fingerprint(labels_dir, image_names)
    for name, info in _load_registry(Path(registry_path)).items():
        if info["hash"] == fp["hash"]:
            return name
    raise ValueError(
        f"현재 검증 라벨(hash {fp['hash']})이 등록된 버전이 아닙니다. "
        f"register_label_version()으로 먼저 등록하세요."
    )
