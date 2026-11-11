from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import URL, create_engine, event, text
from sqlalchemy.engine import Connection, RowMapping

from config import DB_PATH
from models import Base

engine = create_engine(
    URL.create("sqlite", database=str(DB_PATH)),
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)

@event.listens_for(engine, "connect")
def enable_sqlite_foreign_keys(connection: Any, record: Any) -> None:
    connection.execute("PRAGMA foreign_keys=ON")

class DatabaseResult:
    def __init__(self, result: Any):
        self.result = result
        self.rowcount = result.rowcount

    def fetchone(self) -> RowMapping | None:
        row = self.result.fetchone()
        return row._mapping if row is not None else None

    def fetchall(self) -> list[RowMapping]:
        return [row._mapping for row in self.result.fetchall()]

class DatabaseConnection:
    def __init__(self, connection: Connection):
        self.connection = connection

    @staticmethod
    def _bind(statement: str, parameters: tuple | list) -> tuple[str, dict[str, Any]]:
        values = tuple(parameters)
        parts = statement.split("?")
        if len(parts) - 1 != len(values):
            raise ValueError("SQL parameter count does not match the statement.")
        compiled = parts[0]
        bindings: dict[str, Any] = {}
        for index, value in enumerate(values):
            name = f"p{index}"
            compiled += f":{name}{parts[index + 1]}"
            bindings[name] = value
        return compiled, bindings

    def execute(self, statement: str, parameters: tuple | list = ()) -> DatabaseResult:
        compiled, bindings = self._bind(statement, parameters)
        return DatabaseResult(self.connection.execute(text(compiled), bindings))

    def executemany(self, statement: str, parameters: list[tuple]) -> DatabaseResult:
        compiled, _ = self._bind(statement, parameters[0] if parameters else ())
        names = [f"p{index}" for index in range(len(parameters[0]))] if parameters else []
        bindings = [dict(zip(names, values)) for values in parameters]
        return DatabaseResult(self.connection.execute(text(compiled), bindings))

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

@contextmanager
def database() -> Iterator[DatabaseConnection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with engine.begin() as connection:
        yield DatabaseConnection(connection)

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    with database() as con:
        con.execute(
            "INSERT OR IGNORE INTO timer_settings(id, focus_minutes, break_minutes) VALUES(1, 25, 5)"
        )

