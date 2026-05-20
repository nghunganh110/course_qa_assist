"""JSON file-based data layer for Chainlit conversation persistence.

Stores each conversation thread as a separate JSON file under a configurable
directory. Designed for single-user local usage (no authentication overhead,
no concurrent access concerns).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from chainlit.data.base import BaseDataLayer
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

if TYPE_CHECKING:
    from chainlit.element import Element, ElementDict
    from chainlit.step import StepDict


class JsonDataLayer(BaseDataLayer):
    """Minimal JSON file-based data layer for local chat history persistence.

    Each thread is stored as ``{thread_id}.json`` under *storage_dir/threads/*.
    """

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.threads_dir = storage_dir / "threads"
        self.threads_dir.mkdir(parents=True, exist_ok=True)

    # -- helpers ---------------------------------------------------------------

    def _thread_path(self, thread_id: str) -> Path:
        return self.threads_dir / f"{thread_id}.json"

    def _read_thread(self, thread_id: str) -> Optional[ThreadDict]:
        path = self._thread_path(thread_id)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _write_thread(self, thread: ThreadDict) -> None:
        path = self._thread_path(thread["id"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(thread, f, indent=2, default=str, ensure_ascii=False)

    def _ensure_thread(self, thread_id: str) -> ThreadDict:
        thread = self._read_thread(thread_id)
        if thread is None:
            thread = {
                "id": thread_id,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "name": None,
                "userId": None,
                "userIdentifier": "local",
                "tags": None,
                "metadata": {},
                "steps": [],
                "elements": [],
            }
            self._write_thread(thread)
        return thread

    # -- user management (single-user local app) ------------------------------

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        return PersistedUser(
            id="local",
            identifier=identifier,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        return PersistedUser(
            id="local",
            identifier=user.identifier,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )

    # -- feedback (no-op) -----------------------------------------------------

    async def upsert_feedback(self, feedback: Feedback) -> str:
        return feedback.id or str(uuid.uuid4())

    async def delete_feedback(self, feedback_id: str) -> bool:
        return True

    # -- element management ---------------------------------------------------

    async def create_element(self, element: "Element") -> None:
        pass

    async def get_element(
        self, thread_id: str, element_id: str
    ) -> Optional["ElementDict"]:
        return None

    async def delete_element(
        self, element_id: str, thread_id: Optional[str] = None
    ) -> None:
        pass

    # -- step management ------------------------------------------------------

    async def create_step(self, step_dict: "StepDict") -> None:
        thread_id = step_dict.get("threadId")
        if not thread_id:
            return
        thread = self._ensure_thread(thread_id)
        existing_ids = {s["id"] for s in thread.get("steps", [])}
        if step_dict.get("id") not in existing_ids:
            thread.setdefault("steps", []).append(step_dict)
            self._write_thread(thread)

    async def update_step(self, step_dict: "StepDict") -> None:
        thread_id = step_dict.get("threadId")
        if not thread_id:
            return
        thread = self._read_thread(thread_id)
        if not thread:
            return
        steps = thread.get("steps", [])
        for i, s in enumerate(steps):
            if s.get("id") == step_dict.get("id"):
                steps[i] = step_dict
                break
        thread["steps"] = steps
        self._write_thread(thread)

    async def delete_step(self, step_id: str) -> None:
        pass

    # -- thread management ----------------------------------------------------

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        return self._read_thread(thread_id)

    async def get_thread_author(self, thread_id: str) -> str:
        return "local"

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        thread = self._ensure_thread(thread_id)
        if name is not None:
            thread["name"] = name
        if user_id is not None:
            thread["userId"] = user_id
        if metadata is not None:
            thread["metadata"] = metadata
        if tags is not None:
            thread["tags"] = tags
        self._write_thread(thread)

    async def delete_thread(self, thread_id: str) -> None:
        path = self._thread_path(thread_id)
        if path.exists():
            path.unlink()

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        threads: list[ThreadDict] = []
        for path in self.threads_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    thread = json.load(f)
                    if thread.get("steps"):
                        threads.append(thread)
            except (json.JSONDecodeError, OSError):
                continue

        threads.sort(key=lambda t: t.get("createdAt", ""), reverse=True)

        start = 0
        if pagination.cursor:
            for i, t in enumerate(threads):
                if t["id"] == pagination.cursor:
                    start = i + 1
                    break

        page_size = pagination.first or 20
        page = threads[start : start + page_size]
        has_next = (start + page_size) < len(threads)

        return PaginatedResponse(
            data=page,
            pageInfo=PageInfo(
                hasNextPage=has_next,
                startCursor=page[0]["id"] if page else None,
                endCursor=page[-1]["id"] if page else None,
            ),
        )

    # -- misc -----------------------------------------------------------------

    async def get_favorite_steps(self, user_id: str) -> List["StepDict"]:
        return []

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass
