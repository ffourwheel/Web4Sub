import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "backend" / "survey.db"
CSV_PATH = BASE_DIR / "models" / "data" / "cleansing_water_data.csv"

# ── Column definitions ─────────────────────────────────
# All 28 columns from the CSV, preserving original names
TEXT_COLUMNS = [
    "sex", "age", "occupation", "monthly_income", "province",
    "skin_type", "concerns", "acne_level", "consult_influencer",
    "skincare_face_method", "cleansing_method", "use_cleansing_water",
    "cleansing_types_used", "cleansing_type_most_used",
    "cleansing_water_formula", "switch_factors",
    "brands_used", "brand_primary",
]

FLOAT_COLUMNS = [
    "factor_deep_cleansing", "factor_acne_friendly",
    "factor_sensitive_friendly", "factor_no_allergen",
    "factor_hypoallergenic", "factor_moisturizing",
    "factor_low_friction", "factor_nourishment",
    "factor_eye_friendly", "factor_oil_control",
]

ALL_COLUMNS = TEXT_COLUMNS + FLOAT_COLUMNS


# ── Connection helper ──────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ── Table creation ─────────────────────────────────────
def create_tables():
    text_cols = ", ".join([f'"{c}" TEXT' for c in TEXT_COLUMNS])
    float_cols = ", ".join([f'"{c}" REAL' for c in FLOAT_COLUMNS])

    sql = f"""
    CREATE TABLE IF NOT EXISTS survey_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {text_cols},
        {float_cols}
    )
    """
    with get_db() as conn:
        conn.execute(sql)
        conn.commit()
    print(f"Table 'survey_responses' ready in {DB_PATH}")


# ── CSV import ─────────────────────────────────────────
def import_csv():
    import pandas as pd

    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(str(CSV_PATH))
    print(f"Read {len(df)} rows from CSV")

    with get_db() as conn:
        conn.execute("DELETE FROM survey_responses")

        placeholders = ", ".join(["?"] * len(ALL_COLUMNS))
        col_names = ", ".join([f'"{c}"' for c in ALL_COLUMNS])
        sql = f'INSERT INTO survey_responses ({col_names}) VALUES ({placeholders})'

        rows = []
        for _, row in df.iterrows():
            values = []
            for col in ALL_COLUMNS:
                val = row.get(col, None)
                if hasattr(val, 'item'):
                    val = val.item()
                if isinstance(val, float) and (val != val):
                    val = None
                values.append(val)
            rows.append(tuple(values))

        conn.executemany(sql, rows)
        conn.commit()

    print(f"Imported {len(rows)} rows into survey_responses")


# ── Query helpers ──────────────────────────────────────
def fetch_all(query: str, params: tuple = ()) -> list[dict]:
    with get_db() as conn:
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def fetch_one(query: str, params: tuple = ()) -> dict | None:
    with get_db() as conn:
        cursor = conn.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None


def count_rows(where: str = "", params: tuple = ()) -> int:
    sql = "SELECT COUNT(*) as cnt FROM survey_responses"
    if where:
        sql += f" WHERE {where}"
    result = fetch_one(sql, params)
    return result["cnt"] if result else 0


def value_counts(column: str, top_n: int | None = None,
                 where: str = "", params: tuple = ()) -> dict:
    sql = f'''
        SELECT "{column}" as val, COUNT(*) as cnt
        FROM survey_responses
        WHERE "{column}" IS NOT NULL
    '''
    if where:
        sql += f" AND ({where})"
    sql += f' GROUP BY "{column}" ORDER BY cnt DESC'
    if top_n:
        sql += f" LIMIT {top_n}"

    rows = fetch_all(sql, params)
    return {row["val"]: row["cnt"] for row in rows}


def column_mean(column: str, where: str = "", params: tuple = ()) -> float:
    sql = f'SELECT AVG("{column}") as avg_val FROM survey_responses'
    if where:
        sql += f" WHERE {where}"
    result = fetch_one(sql, params)
    return round(result["avg_val"], 2) if result and result["avg_val"] else 0.0


def factor_means(where: str = "", params: tuple = ()) -> dict:
    avgs = ", ".join([f'ROUND(AVG("{c}"), 2) as "{c}"' for c in FLOAT_COLUMNS])
    sql = f"SELECT {avgs} FROM survey_responses"
    if where:
        sql += f" WHERE {where}"
    result = fetch_one(sql, params)
    return dict(result) if result else {}


def total_columns() -> int:
    return len(ALL_COLUMNS)


def missing_values_count() -> int:
    parts = " + ".join([f'(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in ALL_COLUMNS])
    sql = f"SELECT SUM({parts}) as total_nulls FROM survey_responses"
    result = fetch_one(sql)
    return int(result["total_nulls"]) if result and result["total_nulls"] else 0


def get_model_performance() -> list[dict]:
    try:
        return fetch_all("SELECT * FROM model_performance")
    except sqlite3.OperationalError:
        return []

# ── Initialize ─────────────────────────────────────────
def init_db():
    create_tables()
    import_csv()


if __name__ == "__main__":
    init_db()
