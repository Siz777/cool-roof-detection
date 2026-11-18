"""seed 반복 학습·평가와 결과 자동 기록.

- run_experiment   : 한 실험(exp_id)을 여러 seed로 학습 → best.pt를 고정 검증셋으로 평가 → CSV에 한 줄씩 누적
- evaluate_weights : 학습 없이 기존 가중치만 평가해 기록 (예: E1a = 기존 모델을 val_v2로 재평가)
- summarize        : 실험별 평균±표준편차, 기준 실험 대비 Δ, 노이즈(2σ) 대비 판정
- to_markdown      : README에 붙일 표 생성

Colab 런타임이 끊겨도 같은 셀을 다시 실행하면 기록된 seed는 건너뛰고,
중단된 seed는 last.pt에서 이어서 학습한다.

해석 원칙: 누적 실험(E2 = E1 + 해상도 …)의 Δ는 '그 시점 설정에서 추가한 변경의
한계 효과'다. 변경 하나의 독립 기여도는 최종 설정 기준 leave-one-out으로 따로 잰다.
"""
from __future__ import annotations

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

COOL, GENERIC = "cool roof", "generic roof"
METRICS = ["map50", "map75", "map50_95", "cool_map50", "cool_map50_95", "generic_map50_95"]
LOG_COLUMNS = [
    "exp_id", "base_exp", "change", "model", "imgsz", "epochs_max", "patience", "seed",
    "val_version", "best_epoch", "epochs_run", *METRICS, "train_min",
    "run_dir", "weights", "ultralytics", "timestamp",
]


# ---------------------------------------------------------------------------
# 기록
# ---------------------------------------------------------------------------
def _append_row(log_csv: Path, row: dict) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not log_csv.exists()
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def load_log(log_csv: str | Path) -> pd.DataFrame:
    log_csv = Path(log_csv)
    return pd.read_csv(log_csv) if log_csv.exists() else pd.DataFrame(columns=LOG_COLUMNS)


def _done(log_csv: Path, exp_id: str, seed) -> bool:
    df = load_log(log_csv)
    if df.empty:
        return False
    return bool(((df["exp_id"] == exp_id) & (df["seed"].astype(str) == str(seed))).any())


# ---------------------------------------------------------------------------
# 학습 곡선 파싱
# ---------------------------------------------------------------------------
def read_training_curve(run_dir: str | Path) -> dict:
    """results.csv에서 실제 학습 epoch 수, mAP50-95 최고 epoch, 누적 학습 시간을 읽는다."""
    f = Path(run_dir) / "results.csv"
    if not f.exists():
        return {"best_epoch": None, "epochs_run": None, "train_min": None}
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]   # 구버전은 컬럼명 앞에 공백이 있음
    col = "metrics/mAP50-95(B)"
    best = int(df.loc[df[col].idxmax(), "epoch"]) if col in df else None
    t = round(float(df["time"].iloc[-1]) / 60, 1) if "time" in df else None
    return {"best_epoch": best, "epochs_run": int(df["epoch"].iloc[-1]), "train_min": t}


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def _class_metrics(box, names: dict) -> dict:
    """클래스별 mAP50, mAP50-95 (해당 클래스가 검증셋에 없으면 NaN)."""
    idx = {names[i]: i for i in names}
    out = {}
    present = list(box.ap_class_index)
    for key, cname in [("cool", COOL), ("generic", GENERIC)]:
        c = idx.get(cname)
        if c is None or c not in present:
            out[f"{key}_map50"], out[f"{key}_map50_95"] = math.nan, math.nan
            continue
        _, _, ap50, ap = box.class_result(present.index(c))
        out[f"{key}_map50"], out[f"{key}_map50_95"] = float(ap50), float(ap)
    return out


def _validate(weights: Path, data_yaml: str, imgsz: int, project: Path, name: str,
              batch: int = 16, device=None) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=data_yaml, split="val", imgsz=imgsz, batch=batch, plots=False,
                  verbose=False, project=str(project), name=name, exist_ok=True, device=device)
    res = {"map50": float(m.box.map50), "map75": float(m.box.map75), "map50_95": float(m.box.map)}
    res.update(_class_metrics(m.box, model.names))
    res.pop("generic_map50", None)
    return {k: round(v, 4) if isinstance(v, float) and not math.isnan(v) else v for k, v in res.items()}


def check_val_list(data_yaml: str | Path, val_list: str | Path | None) -> None:
    """data.yaml이 가리키는 검증 이미지 폴더가 고정 목록(val_list.txt)과 같은지 확인."""
    if val_list is None:
        return
    import yaml

    from .splits import verify_image_list

    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    val_dir = Path(cfg.get("path", Path(data_yaml).parent)) / cfg["val"]
    verify_image_list(val_dir, val_list)


def _ultralytics_version() -> str:
    try:
        import ultralytics
        return ultralytics.__version__
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run_experiment(
    exp_id: str,
    data_yaml: str | Path,
    *,
    val_version: str,
    log_csv: str | Path,
    project: str | Path,
    change: str = "",
    base_exp: str = "",
    model: str = "yolov8n.pt",
    imgsz: int = 640,
    epochs: int = 100,
    patience: int = 20,
    seeds=(0, 1, 2),
    batch: int = 16,
    device=None,
    val_list: str | Path | None = None,
    **train_kwargs,
) -> pd.DataFrame:
    """exp_id 실험을 seed별로 학습·평가하고 log_csv에 기록한다. 이 실험의 기록을 반환.

    val_list를 주면 학습 전에 검증셋 구성이 고정 목록과 같은지 확인한다.
    """
    from ultralytics import YOLO

    data_yaml, log_csv = str(Path(data_yaml).resolve()), Path(log_csv)
    project = Path(project).resolve()
    check_val_list(data_yaml, val_list)

    for seed in seeds:
        run_name = f"{exp_id}_s{seed}"
        run_dir = project / run_name
        if _done(log_csv, exp_id, seed):
            print(f"⏭️ {run_name}: 이미 기록됨 — 건너뜀")
            continue

        last = run_dir / "weights" / "last.pt"
        t0 = time.time()
        if last.exists():
            print(f"🔁 {run_name}: 중단된 학습 재개 (last.pt)")
            try:
                YOLO(str(last)).train(resume=True)
            except AssertionError as e:   # 이미 끝난 학습이면 Ultralytics가 resume을 거부함
                print(f"   재개 불필요 — {e}")
        else:
            print(f"🚀 {run_name}: 학습 시작 (model={model}, imgsz={imgsz}, epochs≤{epochs}, seed={seed})")
            YOLO(model).train(
                data=data_yaml, imgsz=imgsz, epochs=epochs, patience=patience, batch=batch,
                seed=seed, deterministic=True, project=str(project), name=run_name,
                exist_ok=True, device=device, **train_kwargs,
            )

        best = run_dir / "weights" / "best.pt"
        assert best.exists(), f"{best} 없음 — 학습이 정상 종료되지 않았습니다."
        curve = read_training_curve(run_dir)
        if curve["train_min"] is None:
            curve["train_min"] = round((time.time() - t0) / 60, 1)
        metrics = _validate(best, data_yaml, imgsz, project, f"{run_name}_val", batch, device)

        row = {
            "exp_id": exp_id, "base_exp": base_exp, "change": change, "model": Path(model).stem,
            "imgsz": imgsz, "epochs_max": epochs, "patience": patience, "seed": seed,
            "val_version": val_version, **curve, **metrics, "run_dir": str(run_dir),
            "weights": str(best), "ultralytics": _ultralytics_version(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        _append_row(log_csv, row)
        print(f"📝 {run_name}: mAP50-95={metrics['map50_95']:.4f} | "
              f"cool={metrics['cool_map50_95']:.4f} | best_epoch={curve['best_epoch']}")

    df = load_log(log_csv)
    return df[df["exp_id"] == exp_id].reset_index(drop=True)


def evaluate_weights(
    exp_id: str,
    weights: str | Path,
    data_yaml: str | Path,
    *,
    val_version: str,
    log_csv: str | Path,
    project: str | Path,
    change: str = "",
    base_exp: str = "",
    seed="",
    imgsz: int = 640,
    batch: int = 16,
    device=None,
    val_list: str | Path | None = None,
) -> dict:
    """학습 없이 기존 가중치를 평가만 해서 기록한다 (예: E1a — 측정 보정 효과)."""
    weights, log_csv = Path(weights), Path(log_csv)
    check_val_list(data_yaml, val_list)
    if _done(log_csv, exp_id, seed):
        print(f"⏭️ {exp_id} (seed={seed}): 이미 기록됨 — 건너뜀")
        return {}
    name = f"{exp_id}_s{seed}_val" if seed != "" else f"{exp_id}_val"
    metrics = _validate(weights, str(Path(data_yaml).resolve()), imgsz, Path(project).resolve(),
                        name, batch, device)
    row = {
        "exp_id": exp_id, "base_exp": base_exp, "change": change,
        "model": weights.parent.parent.name, "imgsz": imgsz, "seed": seed,
        "val_version": val_version, **metrics, "weights": str(weights),
        "ultralytics": _ultralytics_version(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    _append_row(log_csv, row)
    print(f"📝 {exp_id}: mAP50-95={metrics['map50_95']:.4f} | cool={metrics['cool_map50_95']:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
def summarize(log_csv: str | Path | pd.DataFrame, metrics=tuple(METRICS),
              noise_ref: str = "B0", k: float = 2.0) -> pd.DataFrame:
    """실험별 평균·표준편차와 기준 실험(base_exp) 대비 Δ, 노이즈 판정을 계산한다.

    판정 기준: |Δ mAP50-95| > k × σ(noise_ref).  σ는 noise_ref 실험의 seed 간 표준편차.
    (n=3 seed의 σ는 그 자체로 불확실하므로 '채택 후보'는 최종 판단이 아니라 1차 선별이다.)
    """
    df = log_csv if isinstance(log_csv, pd.DataFrame) else load_log(log_csv)
    if df.empty:
        return pd.DataFrame()
    order = list(dict.fromkeys(df["exp_id"]))
    meta_cols = ["base_exp", "change", "model", "imgsz", "val_version"]

    rows = []
    for exp in order:
        g = df[df["exp_id"] == exp]
        r = {"exp_id": exp, "n_seeds": len(g)}
        for c in meta_cols:
            r[c] = g[c].iloc[0] if c in g else ""
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce")
            r[f"{m}_mean"] = vals.mean()
            r[f"{m}_std"] = vals.std(ddof=1) if vals.count() > 1 else math.nan
        rows.append(r)
    s = pd.DataFrame(rows).set_index("exp_id")
    for c in ("base_exp", "val_version"):
        s[c] = s[c].fillna("").astype(str).replace("nan", "")

    sigma = s.loc[noise_ref, "map50_95_std"] if noise_ref in s.index else math.nan
    for m in metrics:
        s[f"{m}_delta"] = [
            s.loc[e, f"{m}_mean"] - s.loc[b, f"{m}_mean"] if b in s.index else math.nan
            for e, b in zip(s.index, s["base_exp"])
        ]

    def verdict(exp):
        b = s.loc[exp, "base_exp"]
        if not b or b not in s.index:
            return "기준선"
        d = s.loc[exp, "map50_95_delta"]
        if math.isnan(sigma) or math.isnan(d):
            return "판정 불가(σ 없음)"
        note = " · val 버전 다름" if s.loc[exp, "val_version"] != s.loc[b, "val_version"] else ""
        if d > k * sigma:
            return "채택 후보" + note
        if d < -k * sigma:
            return "악화" + note
        return "노이즈 범위" + note

    s["verdict"] = [verdict(e) for e in s.index]
    s.attrs["sigma_ref"], s.attrs["noise_ref"], s.attrs["k"] = sigma, noise_ref, k
    return s


def _fmt(mean, std, digits=3) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "-"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _fmt_delta(d, digits=3) -> str:
    return "-" if d is None or math.isnan(d) else f"{d:+.{digits}f}"


def to_markdown(summary: pd.DataFrame) -> str:
    """README용 마크다운 표 (평균 ± 표준편차, 기준 대비 Δ, 판정)."""
    head = ("| 실험 | 기준 | 변경 내용 | 검증 라벨 | seed | mAP50 | mAP75 | mAP50-95 | "
            "Cool Roof mAP50-95 | Δ mAP50-95 | 판정 |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for exp, r in summary.iterrows():
        lines.append(
            f"| {exp} | {r['base_exp'] or '-'} | {r['change']} | {r['val_version']} | {r['n_seeds']} | "
            f"{_fmt(r['map50_mean'], r['map50_std'])} | {_fmt(r['map75_mean'], r['map75_std'])} | "
            f"**{_fmt(r['map50_95_mean'], r['map50_95_std'])}** | "
            f"{_fmt(r['cool_map50_95_mean'], r['cool_map50_95_std'])} | "
            f"{_fmt_delta(r['map50_95_delta'])} | {r['verdict']} |"
        )
    sigma = summary.attrs.get("sigma_ref", math.nan)
    if not math.isnan(sigma):
        lines.append("")
        lines.append(f"*판정 기준: |Δ mAP50-95| > {summary.attrs['k']:g}σ, "
                     f"σ = {summary.attrs['noise_ref']}의 seed 간 표준편차 ({sigma:.4f})*")
    return "\n".join(lines)
