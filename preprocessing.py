"""Data cleaning: cancellations, missing values, negative quantities, invalid prices."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PreprocessConfig


def clean_chunk(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    df = df.copy()

    # Remove cancellation invoices (InvoiceNo starting with 'C')
    if cfg.drop_cancelled and "InvoiceNo" in df.columns:
        inv = df["InvoiceNo"].astype(str)
        df = df[~inv.str.upper().str.startswith("C")]

    # Numeric coercion
    for col in ("Quantity", "UnitPrice", "CustomerID"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Drop invalid rows
    req = [c for c in ("CustomerID", "Quantity", "UnitPrice", "InvoiceDate") if c in df.columns]
    if cfg.drop_missing_customer:
        df = df.dropna(subset=req)
    if cfg.drop_negative_quantity:
        df = df[df["Quantity"] > 0]
    if cfg.drop_nonpositive_price:
        df = df[df["UnitPrice"] > 0]

    # Type casting
    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].astype("int64")

    # Amount
    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    # Optional dedupe
    if cfg.drop_duplicates:
        df = df.drop_duplicates(subset=["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "CustomerID"]
                                if {"InvoiceNo", "StockCode", "Quantity", "UnitPrice", "CustomerID"}.issubset(df.columns)
                                else None)

    # Cap extreme monetary outliers at row level
    if "Amount" in df.columns and len(df) > 0:
        cap = df["Amount"].quantile(cfg.outlier_monetary_quantile)
        if np.isfinite(cap) and cap > 0:
            df["Amount"] = df["Amount"].clip(upper=cap)

    return df


def clean_dataframe(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    if df.empty:
        return df
    df = clean_chunk(df, cfg)
    if "InvoiceDate" in df.columns and getattr(df["InvoiceDate"].dt, "tz", None) is None:
        df["InvoiceDate"] = df["InvoiceDate"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward")
        df = df.dropna(subset=["InvoiceDate"])
    return df.sort_values("InvoiceDate").reset_index(drop=True)