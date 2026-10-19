"""
Objective 1 – Data Collection
Extract and consolidate guest reviews with booking, guest and hotel data
from Montra's PostgreSQL operational database.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from config.config import DATABASE_URL, DATA_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)


def get_engine(url: str = DATABASE_URL):
    """Create a SQLAlchemy engine."""
    return create_engine(url, pool_pre_ping=True)


def extract_table(table_name: str, engine=None) -> pd.DataFrame:
    """Extract a single table into a DataFrame."""
    engine = engine or get_engine()
    logger.info("Extracting table: %s", table_name)
    df = pd.read_sql(text(f"SELECT * FROM {table_name}"), engine)
    logger.info("  → %d rows, %d columns", len(df), len(df.columns))
    return df


def extract_all(engine=None) -> dict[str, pd.DataFrame]:
    """Extract all core tables."""
    engine = engine or get_engine()
    tables = ["reviews", "bookings", "guests", "hotels"]
    return {t: extract_table(t, engine) for t in tables}


def build_consolidated_dataset(
    reviews: Optional[pd.DataFrame] = None,
    bookings: Optional[pd.DataFrame] = None,
    guests: Optional[pd.DataFrame] = None,
    hotels: Optional[pd.DataFrame] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Join reviews with booking, guest and hotel context.

    Primary key for reviews is review_id.
    Links:
      reviews.booking_id  → bookings.booking_id
      reviews.guest_id    → guests.guest_id
      reviews.hotel_id    → hotels.hotel_id
    """
    if any(x is None for x in (reviews, bookings, guests, hotels)):
        data = extract_all(engine)
        reviews = reviews if reviews is not None else data["reviews"]
        bookings = bookings if bookings is not None else data["bookings"]
        guests = guests if guests is not None else data["guests"]
        hotels = hotels if hotels is not None else data["hotels"]

    logger.info("Building consolidated guest-experience dataset…")

    # Rename overlapping id columns on the dimension tables to avoid clashes
    bookings = bookings.add_prefix("bk_").rename(columns={"bk_booking_id": "booking_id"})
    guests = guests.add_prefix("gt_").rename(columns={"gt_guest_id": "guest_id"})
    hotels = hotels.add_prefix("ht_").rename(columns={"ht_hotel_id": "hotel_id"})

    df = (
        reviews
        .merge(bookings, on="booking_id", how="left")
        .merge(guests, on="guest_id", how="left")
        .merge(hotels, on="hotel_id", how="left")
    )

    # Parse dates
    for col in ["review_date", "bk_check_in_date", "bk_check_out_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info("Consolidated shape: %s", df.shape)
    return df


def _save_df(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as parquet (preferred) or pickle fallback."""
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_pickle(path.with_suffix(".pkl"))


def _load_df(path_stem: Path) -> pd.DataFrame:
    """Load DataFrame from parquet or pickle."""
    pq = path_stem.with_suffix(".parquet")
    pkl = path_stem.with_suffix(".pkl")
    if pq.exists() and pq.stat().st_size > 0:
        return pd.read_parquet(pq)
    if pkl.exists() and pkl.stat().st_size > 0:
        return pd.read_pickle(pkl)
    raise FileNotFoundError(f"No data file found for {path_stem}")


def save_raw_extracts(data: dict[str, pd.DataFrame], out_dir: Path = DATA_DIR) -> None:
    """Persist raw extracts for reproducibility."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        path = out_dir / f"raw_{name}"
        _save_df(df, path)
        logger.info("Saved %s (%d rows)", name, len(df))


def load_or_extract(
    force_refresh: bool = False,
    out_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Load consolidated dataset from disk if available, otherwise extract from DB.
    Returns (consolidated_df, raw_tables_dict).
    """
    tables = ["reviews", "bookings", "guests", "hotels"]
    consolidated_stem = out_dir / "consolidated_reviews"
    raw_stems = {t: out_dir / f"raw_{t}" for t in tables}

    def _exists(stem: Path) -> bool:
        return any(
            (stem.with_suffix(s).exists() and stem.with_suffix(s).stat().st_size > 0)
            for s in (".parquet", ".pkl")
        )

    if not force_refresh and _exists(consolidated_stem) and all(_exists(s) for s in raw_stems.values()):
        logger.info("Loading cached extracts from %s", out_dir)
        raw = {t: _load_df(s) for t, s in raw_stems.items()}
        consolidated = _load_df(consolidated_stem)
        return consolidated, raw

    raw = extract_all()
    save_raw_extracts(raw, out_dir)
    consolidated = build_consolidated_dataset(**raw)
    _save_df(consolidated, consolidated_stem)
    logger.info("Saved consolidated dataset (%d rows)", len(consolidated))
    return consolidated, raw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df, raw = load_or_extract(force_refresh=True)
    print(df.head())
    print(df.columns.tolist())
    print(df.shape)
