import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS footprints (
    consumer     TEXT NOT NULL,
    key          TEXT NOT NULL,
    footprint_mb INTEGER NOT NULL,
    PRIMARY KEY (consumer, key)
);
"""


class FootprintStore:
    """Measured VRAM footprints, so estimates self-correct after first load.

    Plain SQLite rows a human can read and delete — there is no hand-maintained
    size table anywhere in the codebase to go stale.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get(self, consumer: str, key: str) -> int | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT footprint_mb FROM footprints WHERE consumer = ? AND key = ?",
                (consumer, key),
            ).fetchone()
        return int(row[0]) if row else None

    def record(self, consumer: str, key: str, footprint_mb: int) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO footprints (consumer, key, footprint_mb) VALUES (?, ?, ?) "
                "ON CONFLICT(consumer, key) DO UPDATE SET footprint_mb = excluded.footprint_mb",
                (consumer, key, footprint_mb),
            )

    def all(self) -> dict[tuple[str, str], int]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT consumer, key, footprint_mb FROM footprints"
            ).fetchall()
        return {(r[0], r[1]): int(r[2]) for r in rows}
