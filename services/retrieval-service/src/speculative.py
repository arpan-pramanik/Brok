import asyncio
import time
from typing import Optional


class SpeculativeRetriever:
    def __init__(self, debounce_ms: int = 300):
        self.debounce_ms = debounce_ms
        self._pending_task: Optional[asyncio.Task] = None
        self._last_query: str = ""
        self._latest_result: Optional[dict] = None

    async def on_partial_transcript(self, text: str, retrieve_fn) -> Optional[dict]:
        if text == self._last_query or len(text.split()) < 3:
            return None
        self._last_query = text
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
        self._pending_task = asyncio.create_task(self._debounced_retrieve(text, retrieve_fn))
        return None

    async def _debounced_retrieve(self, text: str, retrieve_fn):
        try:
            await asyncio.sleep(self.debounce_ms / 1000)
            result = await asyncio.get_event_loop().run_in_executor(None, retrieve_fn, text)
            self._latest_result = result
        except asyncio.CancelledError:
            pass

    def get_latest_result(self) -> Optional[dict]:
        return self._latest_result

    def clear(self):
        self._latest_result = None
        self._last_query = ""
