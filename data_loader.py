"""Chunked, robust data loading for Online Retail II and similar CSV/XLSX files."""
from __future__ import annotations
import re
from typing import Iterator, Optional
import pandas as pd
from .config import DataConfig


def _clean_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).replace("\ufeff", "").strip())


def _normalize_key(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


CANONICAL_CANDIDATES = {
    "InvoiceNo": ["InvoiceNo", "Invoice", "Invoice Number", "InvoiceNo."],
    "Quantity": ["Quantity", "Qty"],
    "InvoiceDate": ["InvoiceDate", "Invoice Date", "Date"],
    "UnitPrice": ["UnitPrice", "Unit Price", "Price"],
    "CustomerID": ["CustomerID", "Customer ID", "Customer Id", "Customer"],
    "StockCode": ["StockCode", "Stock Code", "SKU"],
    "Description": ["Description", "Item"],
    "Country": ["Country"],
}


def _resolve_columns(columns: list[str]) -> dict[str, Optional[str]]:
    norm = {_normalize_key(c): c for c in columns}
    resolved: dict[str, Optional[str]] = {}
    for canonical, candidates in CANONICAL_CANDIDATES.items():
        resolved[canonical] = None
        for cand in candidates:
            key = _normalize_key(cand)
            if key in norm:
                resolved[canonical] = norm[key]
                break
    return resolved


def _read_header(config: DataConfig) -> tuple[list[str], str, str]:
    """Detect working encoding/separator by trying candidates on a small sample."""
    for enc in config.encoding_candidates:
        for sep in config.sep_candidates:
            try:
                sample = pd.read_csv(
                    config.path, encoding=enc, sep=sep,
                    nrows=5, engine="python", on_bad_lines="skip",
                )
                if sample.shape[1] >= 5:
                    return list(sample.columns), enc, sep
            except Exception:
                continue
    raise ValueError("Could not detect encoding/separator for the dataset.")


def iter_chunks(config: DataConfig) -> Iterator[pd.DataFrame]:
    """Yield normalized DataFrame chunks with canonical column names."""
    columns, encoding, sep = _read_header(config)
    resolved = _resolve_columns(columns)
    missing = [k for k, v in resolved.items() if v is None and k in (
        "InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID")]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Found: {columns}")

    rename_map = {v: k for k, v in resolved.items() if v is not None}

    reader = pd.read_csv(
        config.path,
        encoding=encoding, sep=sep, engine="c",
        chunksize=config.chunk_size, low_memory=False,
        on_bad_lines="skip",
    )
    for chunk in reader:
        chunk.columns = [_clean_col(c) for c in chunk.columns]
        chunk = chunk.rename(columns=rename_map)
        yield chunk


def load_all(config: DataConfig) -> pd.DataFrame:
    """Load full dataset into memory (concatenates chunks)."""
    parts = list(iter_chunks(config))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)