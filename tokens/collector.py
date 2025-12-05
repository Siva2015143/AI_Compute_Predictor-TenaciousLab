# tenaciouslab/tokens/collector.py
from __future__ import annotations

import json
import logging
import zipfile
from hashlib import sha1
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd
from tqdm import tqdm
from datetime import datetime

logger = logging.getLogger("tokens.collector")
if not logger.handlers:
    # basic fallback logging when module run directly
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


def _to_path(p: Union[str, Path]) -> Path:
    return Path(p) if not isinstance(p, Path) else p


def _safe_load_jsonl(path: Path) -> List[dict]:
    """Safely read a .jsonl file and return list of dict records."""
    recs: List[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        recs.append(obj)
                except json.JSONDecodeError:
                    # skip malformed lines
                    logger.debug("Skipping malformed JSON at %s:%d", path, i)
    except Exception as e:
        logger.warning("Could not read %s — %s", path, e)
    return recs


def _extract_jsonl_from_zip(zip_path: Path, dest_dir: Path) -> List[Path]:
    """
    Extract any top-level .jsonl files from an export bundle into dest_dir.
    Returns list of extracted file paths (skips files that already exist).
    """
    extracted: List[Path] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                if member.lower().endswith(".jsonl"):
                    # write only if not exists to avoid overwriting
                    target_name = Path(member).name
                    target_path = dest_dir / target_name
                    if target_path.exists():
                        logger.debug("Skipping existing extracted file: %s", target_path)
                        continue
                    try:
                        # extract to a temp file path inside dest_dir
                        z.extract(member, dest_dir)
                        # If the member contains subdirs, z.extract will create those — move top-level file to dest_dir root
                        src_extracted = dest_dir / member
                        if src_extracted.exists() and src_extracted != target_path:
                            # move to root and remove intermediate dirs if any
                            src_extracted.replace(target_path)
                        extracted.append(target_path)
                        logger.info("Extracted %s from %s -> %s", member, zip_path.name, target_path.name)
                    except Exception as e:
                        logger.warning("Failed to extract %s from %s: %s", member, zip_path, e)
    except zipfile.BadZipFile:
        logger.warning("Bad ZIP file (skipping): %s", zip_path)
    except Exception as e:
        logger.warning("Error extracting %s: %s", zip_path, e)
    return extracted


def _discover_token_files(data_dir: Path, max_files: Optional[int] = None) -> List[Path]:
    """Return sorted list of jsonl files under data_dir, optionally limited by max_files."""
    files = sorted([p for p in data_dir.glob("*.jsonl") if p.is_file()])
    if max_files is not None:
        files = files[:max_files]
    return files


def _compute_record_key(rec: dict) -> str:
    """
    Create deterministic key for deduplication:
    Prefer unique_key > trial_id > sha1(raw_preview)
    """
    if rec.get("unique_key"):
        return str(rec["unique_key"])
    if rec.get("trial_id"):
        return f"trial:{rec.get('trial_id')}"
    # fallback: hash of raw_preview or full json
    preview = rec.get("raw_preview") or json.dumps(rec, sort_keys=True)
    return "hash:" + sha1(preview.encode("utf-8")).hexdigest()[:12]


def _coerce_numeric_series(s: pd.Series, fallback: float = 0.0) -> pd.Series:
    """Try to coerce a series to numeric, fillna with fallback."""
    s_num = pd.to_numeric(s, errors="coerce")
    return s_num.fillna(fallback)



def _resolve_token_path(data_dir: Union[str, Path, None]) -> Path:
    """
    Resolve compute_tokens path priority:
    1️⃣ User-provided arg
    2️⃣ Ai_Agent external folder on Desktop
    3️⃣ Local fallback in Tenacious_Lab
    """
    if data_dir:
        p = Path(data_dir)
        if p.exists():
            return p

    ai_agent_dir = Path.home() / "Desktop" / "Ai_Agent" / "neuraloperator" / "data" / "compute_tokens"
    if ai_agent_dir.exists():
        logger.info(f"📂 Using external Ai_Agent token directory: {ai_agent_dir}")
        return ai_agent_dir

    local_dir = Path(__file__).resolve().parents[1] / "data" / "compute_tokens"
    logger.info(f"📂 Using local fallback token directory: {local_dir}")
    return local_dir

def collect_tokens(
    data_dir: Union[str, Path, None] = None,  # <-- default None
    save_merged: bool = False,
    extract_from_webui_exports: bool = True,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Collect and merge token JSONL files into a single pandas.DataFrame.

    Args:
        data_dir: path to compute_tokens folder (str or Path).
        save_merged: if True, save a timestamped merged_tokens_*.jsonl in data_dir.
        extract_from_webui_exports: when True attempts to extract jsonl files from ../web-ui/exports zip bundles.
        max_files: optional limit to number of files to read (useful for debugging/autopipeline).

    Returns:
        pd.DataFrame with normalized columns. (Does NOT raise on minor read errors.)
    """
    data_dir = _resolve_token_path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Token directory not found: {data_dir}")

    # optional extraction from sibling web-ui exports (idempotent)
    if extract_from_webui_exports:
        # try to locate web-ui exports relative to project root (common layout)
        project_root = Path(__file__).resolve().parents[1]
        webui_exports = project_root.parent / "web-ui" / "exports"
        if webui_exports.exists():
            for zip_path in sorted(webui_exports.glob("export_bundle_*.zip")):
                _extract_jsonl_from_zip(zip_path, data_dir)
        else:
            logger.debug("No web-ui/exports found at %s — skipping zip extraction", webui_exports)

    files = _discover_token_files(data_dir, max_files=max_files)
    if not files:
        raise FileNotFoundError(f"No .jsonl token files found in {data_dir}")

    logger.info("Loading %d token files from %s", len(files), data_dir)

    records: List[dict] = []
    for p in tqdm(files, desc="Loading token files", disable=False):
        recs = _safe_load_jsonl(p)
        # attach source filename for traceability if missing
        for r in recs:
            if "_source_file" not in r:
                r["_source_file"] = p.name
        records.extend(recs)

    if not records:
        raise RuntimeError("No usable records found after parsing token files.")

    # Build DataFrame
    df = pd.DataFrame(records)

    # --- Ensure canonical columns exist (without deleting extras) ---
    canonical_cols = [
        "unique_key",
        "trial_id",
        "call_id",
        "timestamp",
        "model_name",
        "total_GFLOPs",
        "total_tokens",
        "loss",
        "latency_s",
        "amplification",
        "b_hat",
        "notes",
        "_source_file",
    ]
    for c in canonical_cols:
        if c not in df.columns:
            df[c] = None

    # --- Deduplicate using stable key ---
    df["_collector_key"] = df.apply(lambda r: _compute_record_key(r.to_dict()), axis=1)
    df = df.drop_duplicates(subset=["_collector_key"], keep="last").reset_index(drop=True)
    df.drop(columns=["_collector_key"], inplace=True)
    
    # --- Clean empty strings in object columns before numeric coercion ---
    # Replace blank strings ("", " ", etc.) with None so numeric conversion doesn't fail later
    df = df.replace(r"^\s*$", None, regex=True)


    # --- Coerce and fill numeric columns safely ---
    numeric_cols = ["total_GFLOPs", "total_tokens", "loss", "latency_s", "amplification", "b_hat"]
    for col in numeric_cols:
        if col in df.columns:
            # If column is object, try to coerce
            df[col] = _coerce_numeric_series(df[col], fallback=0.0)

    # enforce reasonable defaults
    if "amplification" in df.columns:
        df["amplification"] = df["amplification"].replace(0.0, 1.0).fillna(1.0)
    if "b_hat" in df.columns:
        df["b_hat"] = df["b_hat"].replace(0.0, 1.0).fillna(1.0)

    # parse timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Final: reset index and report
    df = df.reset_index(drop=True)
    logger.info("Collected %d unique token records (files=%d)", len(df), len(files))

    # Optionally save merged file (useful if caller expects merged jsonl)
    if save_merged:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = data_dir / f"merged_tokens_{ts}.jsonl"
        try:
            # write records as lines
            with out_path.open("w", encoding="utf-8") as fh:
                for rec in df.to_dict(orient="records"):
                    fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
            logger.info("Saved merged tokens -> %s", out_path)
        except Exception as e:
            logger.warning("Failed to write merged file %s: %s", out_path, e)

    return df

if __name__ == "__main__":
    # Explicit paths
    ai_agent_tokens = Path("C:/Users/sivam/Desktop/Ai_Agent/neuraloperator/data/compute_tokens")
    tenaciouslab_tokens = Path("C:/Users/sivam/Desktop/Tenacious_Lab/tenaciouslab/data/compute_tokens")

    # Collect tokens from Ai_Agent folder
    df = collect_tokens(
        data_dir=ai_agent_tokens,
        save_merged=True,  # saves merged_tokens_*.jsonl in Ai_Agent folder
        extract_from_webui_exports=False  # skip zip extraction
    )

    # Move merged file explicitly to Tenacious_Lab folder
    import shutil
    merged_file = list(ai_agent_tokens.glob("merged_tokens_*.jsonl"))[-1]  # latest merged file
    target_file = tenaciouslab_tokens / merged_file.name
    shutil.copy2(merged_file, target_file)
    print(f"Merged tokens copied to Tenacious_Lab: {target_file}")