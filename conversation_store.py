"""SQLite persistence for Streamlit chat conversations."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path


def load_conversations(database_path: Path) -> dict[str, dict]:
    """Load saved conversations, returning an empty collection before first use."""
    if not database_path.exists():
        return {}

    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    updated REAL NOT NULL
                )
                """
            )
            rows = connection.execute(
                """
                SELECT conversation_id, title, messages_json, updated
                FROM conversations
                ORDER BY updated DESC
                """
            ).fetchall()

    return {
        row["conversation_id"]: {
            "title": row["title"],
            "messages": json.loads(row["messages_json"]),
            "updated": row["updated"],
        }
        for row in rows
    }


def save_conversation(
    database_path: Path,
    conversation_id: str,
    conversation: dict,
) -> None:
    """Insert or update a conversation in a single SQLite transaction."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    updated REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id, title, messages_json, updated
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    title = excluded.title,
                    messages_json = excluded.messages_json,
                    updated = excluded.updated
                """,
                (
                    conversation_id,
                    conversation["title"],
                    json.dumps(conversation["messages"], ensure_ascii=False),
                    conversation["updated"],
                ),
            )


def delete_conversations(database_path: Path) -> None:
    """Delete all saved conversations while keeping the database reusable."""
    if not database_path.exists():
        return

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute("DELETE FROM conversations")
