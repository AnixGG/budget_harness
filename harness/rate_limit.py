"""Shared request spacing for the event's 60 RPM cap; never contains credentials."""
import asyncio
import fcntl
import json
import math
from pathlib import Path
import time

from .accounting import StopBudget


class RequestPacer:
    def __init__(self, path, rpm=60, *, clock=time.time, sleeper=asyncio.sleep):
        if not isinstance(rpm, int) or isinstance(rpm, bool) or not 1 <= rpm <= 60:
            raise ValueError('Event request rate must be an integer in [1,60]')
        self.path = Path(path)
        self.interval = 60 / rpm + .01
        self.clock, self.sleeper = clock, sleeper

    async def acquire(self, *, seconds_left=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Reserve a slot under a separate stable lock. Waiting never holds the lock.
        with self.path.with_suffix(self.path.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {}
            current = self.clock()
            next_at = state.get('next_at', current)
            if not isinstance(next_at, (float,int)) or not math.isfinite(next_at):
                raise ValueError('Invalid pacing state; do not silently reset it')
            scheduled = max(current, next_at)
            delay = scheduled-current
            if seconds_left is not None and delay >= seconds_left:
                raise StopBudget('Wall deadline before next allowed request slot')
            from .core import atomic_json
            atomic_json(self.path, {'schema_version':1, 'next_at':scheduled+self.interval})
        if delay:
            await self.sleeper(delay)
        return delay
