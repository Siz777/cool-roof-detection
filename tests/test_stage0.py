"""0단계 유틸리티 테스트 — 합성 데이터와 가짜 ultralytics 모듈만 사용 (GPU·원본 데이터 불필요).

실행: python -m pytest tests -q
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_check as dc  # noqa: E402
from src import dedup, splits  # noqa: E402
from src import experiment as ex  # noqa: E402

NAMES = ["cool roof", "generic roof"]


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def make_split(root: Path, split: str, items: dict[str, str | None], size=32, seed=0):
    """items: {stem: 라벨 텍스트 또는 None(라벨 없음)}"""
    rng = np.random.default_rng(seed)
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for stem, label in items.items():
        arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / split / "images" / f"{stem}.jpg")
        if label is not None:
            (root / split / "labels" / f"{stem}.txt").write_text(label)
    return root / split


# ---------------------------------------------------------------------------
# 클래스 매핑
# ---------------------------------------------------------------------------
def test_mapping_identity_with_name_variants():
    m = dc.build_class_mapping(["Cool-Roof", "generic_roof"], NAMES)
    assert m == {0: 0, 1: 1} and dc.is_identity(m)


def test_mapping_detects_swapped_order():
    m = dc.build_class_mapping(["generic roof", "cool roof"], {0: "cool roof", 1: "generic roof"})
    assert m == {0: 1, 1: 0} and not dc.is_identity(m)
    assert "ID 변경 필요" in dc.describe_mapping(["generic roof", "cool roof"], NAMES, m)


def test_mapping_unmatched_raises_and_manual_resolves():
    with pytest.raises(ValueError):
        dc.build_class_mapping(["white roof", "roof"], NAMES)
    m = dc.build_class_mapping(["white roof", "roof"], NAMES,
                               manual={"white roof": "cool roof", "roof": "generic roof"})
    assert m == {0: 0, 1: 1}


def test_read_class_names_yaml_and_objnames(tmp_path):
    (tmp_path / "a.yaml").write_text("names:\n  - generic roof\n  - cool roof\nnc: 2\n")
    (tmp_path / "b.yaml").write_text("names:\n  1: generic roof\n  0: cool roof\n")
    (tmp_path / "obj.names").write_text("cool roof\ngeneric roof\n\n")
    assert dc.read_class_names(tmp_path / "a.yaml") == ["generic roof", "cool roof"]
    assert dc.read_class_names(tmp_path / "b.yaml") == NAMES
    assert dc.read_class_names(tmp_path / "obj.names") == NAMES


def test_copy_labels_remaps_only_class_column(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n1 0.5 0.5 0.2 0.2\n")
    n = dc.copy_labels(src, tmp_path / "dst", {0: 1, 1: 0})
    assert n == 1
    assert (tmp_path / "dst" / "a.txt").read_text() == \
        "1 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n0 0.5 0.5 0.2 0.2\n"


# ---------------------------------------------------------------------------
# 정합성 검사
# ---------------------------------------------------------------------------
def test_check_split_finds_each_problem(tmp_path):
    d = make_split(tmp_path, "train", {
        "ok_bbox": "0 0.5 0.5 0.2 0.2\n",
        "ok_obb": "1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n",
        "no_label": None,
        "empty": "",
        "bad_cols": "0 0.5 0.5 0.2\n",
        "oob": "1 0.5 0.5 1.4 0.2\n",
        "bad_cls": "2 0.5 0.5 0.2 0.2\n",
        "dup": "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n",
    })
    (d / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    r = dc.check_split(d, num_classes=2)
    assert r.images_without_label == ["no_label"]
    assert r.labels_without_image == ["orphan"]
    assert r.empty_labels == ["empty"]
    assert len(r.bad_lines) == 1 and r.bad_lines[0].startswith("bad_cols")
    assert r.out_of_range == ["oob:1"]
    assert r.unknown_class == ["bad_cls:1 cls=2"]
    assert r.duplicate_lines == ["dup (x2)"]
    assert r.format_counts == {"bbox": 6, "polygon": 1}
    assert r.n_errors == 5
    assert "오류 5건" in r.summary(NAMES)


def test_check_dataset_skips_unlabeled_test(tmp_path, capsys):
    make_split(tmp_path, "train", {"a": "0 0.5 0.5 0.2 0.2\n"})
    make_split(tmp_path, "test", {"x": None})
    reps = dc.check_dataset(tmp_path, NAMES)
    assert set(reps) == {"train"} and reps["train"].n_errors == 0
    assert "추론 전용" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 중복 탐지
# ---------------------------------------------------------------------------
def _smooth_image(seed, size=64):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (6, 6, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((size, size), Image.Resampling.BICUBIC)


def test_dedup_finds_rotated_and_flipped_copies(tmp_path):
    tr, va = tmp_path / "train", tmp_path / "val"
    tr.mkdir()
    va.mkdir()
    base = _smooth_image(1)
    base.save(tr / "orig.png")
    _smooth_image(2).save(tr / "other.png")
    base.rotate(90).save(va / "rot90.png")
    base.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(va / "flip.png")
    _smooth_image(3).save(va / "unrelated.png")

    dups = dedup.find_cross_split_duplicates({"train": tr, "val": va}, threshold=4)
    found = {(p.image_a, p.image_b) for p in dups}
    assert ("orig.png", "rot90.png") in found
    assert ("orig.png", "flip.png") in found
    assert not any("unrelated" in b or "other" in a for a, b in found)
    assert "근접 중복" in dedup.report_duplicates(dups)
    assert "없음" in dedup.report_duplicates([])


# ---------------------------------------------------------------------------
# 검증셋 고정 · 라벨 버전
# ---------------------------------------------------------------------------
def test_image_list_freeze_and_verify(tmp_path):
    d = make_split(tmp_path, "val", {"a": "0 0.5 0.5 0.1 0.1\n", "b": "1 0.5 0.5 0.1 0.1\n"})
    lst = tmp_path / "splits" / "val_list.txt"
    assert splits.save_image_list(d / "images", lst) == ["a.jpg", "b.jpg"]
    splits.save_image_list(d / "images", lst)          # 두 번째는 검증만
    Image.new("RGB", (8, 8)).save(d / "images" / "c.jpg")
    with pytest.raises(AssertionError, match="추가 1장"):
        splits.verify_image_list(d / "images", lst)


def test_label_versions_are_immutable(tmp_path):
    d = make_split(tmp_path, "val", {"a": "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n"})
    reg = tmp_path / "splits" / "label_versions.json"
    v1 = splits.register_label_version(d / "labels", reg, "val_v1")
    assert v1["n_objects"] == 2
    # 줄 순서·공백만 다르면 같은 내용으로 본다
    (d / "labels" / "a.txt").write_text("1  0.2 0.2 0.1 0.1\n0 0.5 0.5 0.1 0.1\n")
    assert splits.identify_label_version(d / "labels", reg) == "val_v1"
    # 내용이 바뀌면 같은 이름으로는 등록 불가, 새 이름으로 등록
    (d / "labels" / "a.txt").write_text("0 0.5 0.5 0.12 0.1\n")
    with pytest.raises(ValueError, match="이미 다른 내용"):
        splits.register_label_version(d / "labels", reg, "val_v1")
    with pytest.raises(ValueError, match="등록된 버전이 아닙니다"):
        splits.identify_label_version(d / "labels", reg)
    splits.register_label_version(d / "labels", reg, "val_v2")
    assert splits.identify_label_version(d / "labels", reg) == "val_v2"


# ---------------------------------------------------------------------------
# 실험 기록 · 요약
# ---------------------------------------------------------------------------
def test_read_training_curve_handles_padded_columns(tmp_path):
    pd.DataFrame({"                  epoch": [1, 2, 3],
                  "   metrics/mAP50-95(B)": [0.3, 0.55, 0.5],
                  "time": [60, 120, 180]}).to_csv(tmp_path / "results.csv", index=False)
    c = ex.read_training_curve(tmp_path)
    assert c == {"best_epoch": 2, "epochs_run": 3, "train_min": 3.0}


def _log(rows):
    base = {c: "" for c in ex.LOG_COLUMNS}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_summarize_delta_and_verdicts():
    def r(exp, base, seed, v, val="val_v1"):
        return {"exp_id": exp, "base_exp": base, "seed": seed, "val_version": val,
                "map50": v + .2, "map75": v, "map50_95": v, "cool_map50": v,
                "cool_map50_95": v - .1, "generic_map50_95": v + .05}
    df = _log([
        r("B0", "", 0, .54), r("B0", "", 1, .55), r("B0", "", 2, .56),      # σ = 0.01
        r("E1", "B0", 0, .60), r("E1", "B0", 1, .61), r("E1", "B0", 2, .62),
        r("E2", "E1", 0, .615),                                            # Δ ≈ 0.005
        r("E3", "E2", 0, .58, "val_v2"),                                   # Δ ≈ -0.035
    ])
    s = ex.summarize(df)
    assert math.isclose(s.attrs["sigma_ref"], 0.01, abs_tol=1e-9)
    assert math.isclose(s.loc["E1", "map50_95_delta"], 0.06, abs_tol=1e-9)
    assert s.loc["B0", "verdict"] == "기준선"
    assert s.loc["E1", "verdict"] == "채택 후보"
    assert s.loc["E2", "verdict"] == "노이즈 범위"
    assert s.loc["E3", "verdict"] == "악화 · val 버전 다름"
    md = ex.to_markdown(s)
    assert "| E1 | B0 |" in md and "**0.610 ± 0.010**" in md and "+0.060" in md
    assert md.splitlines()[1].count("---") == 11


def test_class_metrics_uses_ap_class_index():
    class Box:
        ap_class_index = np.array([1])            # cool roof(0)가 검증셋에 없는 경우
        def class_result(self, i):
            return (0.8, 0.7, 0.9, 0.6)
    m = ex._class_metrics(Box(), {0: "cool roof", 1: "generic roof"})
    assert math.isnan(m["cool_map50_95"]) and m["generic_map50_95"] == 0.6


# ---------------------------------------------------------------------------
# run_experiment — 가짜 ultralytics로 학습 루프·재개·건너뛰기 검증
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_ultralytics(monkeypatch):
    calls = []

    class Box:
        map50, map75, map = 0.82, 0.6, 0.55
        ap_class_index = np.array([0, 1])
        def class_result(self, i):
            return [(0.7, 0.6, 0.73, 0.48), (0.8, 0.9, 0.91, 0.61)][i]

    class YOLO:
        names = {0: "cool roof", 1: "generic roof"}

        def __init__(self, w):
            self.w = w

        def train(self, **kw):
            calls.append(("train", self.w, kw))
            if kw.get("resume"):
                raise AssertionError("training to 100 epochs is finished, nothing to resume")
            run = Path(kw["project"]) / kw["name"]
            (run / "weights").mkdir(parents=True, exist_ok=True)
            (run / "weights" / "best.pt").write_text("x")
            (run / "weights" / "last.pt").write_text("x")
            pd.DataFrame({"epoch": [1, 2], "metrics/mAP50-95(B)": [0.5, 0.55],
                          "time": [30, 60]}).to_csv(run / "results.csv", index=False)

        def val(self, **kw):
            calls.append(("val", self.w, kw))
            return types.SimpleNamespace(box=Box())

    mod = types.ModuleType("ultralytics")
    mod.YOLO, mod.__version__ = YOLO, "fake"
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return calls


def test_run_experiment_trains_logs_and_skips(tmp_path, fake_ultralytics):
    log, proj, yaml_ = tmp_path / "exp.csv", tmp_path / "runs", tmp_path / "d.yaml"
    yaml_.write_text("names: {0: cool roof, 1: generic roof}\n")
    df = ex.run_experiment("B0", yaml_, val_version="val_v1", log_csv=log, project=proj,
                           change="재학습", seeds=(0, 1))
    assert list(df["seed"]) == [0, 1]
    assert df.loc[0, "cool_map50_95"] == 0.48 and df.loc[0, "best_epoch"] == 2
    trains = [c for c in fake_ultralytics if c[0] == "train"]
    assert trains[0][2]["seed"] == 0 and trains[0][2]["deterministic"] is True

    # 다시 실행하면 기록된 seed는 건너뛴다
    n_before = len(fake_ultralytics)
    ex.run_experiment("B0", yaml_, val_version="val_v1", log_csv=log, project=proj, seeds=(0, 1))
    assert len(fake_ultralytics) == n_before

    # 학습은 끝났지만 기록 전에 끊긴 seed → resume 거부를 처리하고 평가·기록만 수행
    run = proj / "B0_s2" / "weights"
    run.mkdir(parents=True)
    (run / "last.pt").write_text("x")
    (run / "best.pt").write_text("x")
    df = ex.run_experiment("B0", yaml_, val_version="val_v1", log_csv=log, project=proj, seeds=(2,))
    assert len(df) == 3 and fake_ultralytics[-2][2] == {"resume": True}

    # 평가 전용 기록 (E1a)
    ex.evaluate_weights("E1a", proj / "B0_s0" / "weights" / "best.pt", yaml_,
                        val_version="val_v2", log_csv=log, project=proj, base_exp="B0", seed=0)
    s = ex.summarize(log)
    assert list(s.index) == ["B0", "E1a"] and s.loc["B0", "n_seeds"] == 3


def test_run_experiment_refuses_changed_val_set(tmp_path, fake_ultralytics):
    root = tmp_path / "datasets"
    make_split(root, "val", {"a": "0 0.5 0.5 0.1 0.1\n"})
    lst = tmp_path / "val_list.txt"
    splits.save_image_list(root / "val" / "images", lst)
    yaml_ = tmp_path / "d.yaml"
    yaml_.write_text(f"path: {root}\nval: val/images\nnames: {{0: cool roof, 1: generic roof}}\n")
    kw = dict(val_version="val_v1", log_csv=tmp_path / "e.csv", project=tmp_path / "runs", val_list=lst)
    ex.run_experiment("B0", yaml_, seeds=(0,), **kw)          # 일치 → 정상 진행
    Image.new("RGB", (8, 8)).save(root / "val" / "images" / "new.jpg")
    with pytest.raises(AssertionError, match="고정 목록과 다릅니다"):
        ex.run_experiment("B0", yaml_, seeds=(1,), **kw)
