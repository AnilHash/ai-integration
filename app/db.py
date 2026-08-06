import asyncpg
import os


_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=os.environ["DATABASE_URL"], min_size=1, max_size=5
        )
    return _pool


async def init_request_log_table():
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS request_log (
                           id SERIAL PRIMARY KEY,
                           trace_id  TEXT NOT NULL,
                           user_id TEXT NOT NULL,
                           query TEXT NOT NULL,
                           query_length INT NOT NULL,
                           answer_length INT,
                           cites_source BOOLEAN,
                           reasonable_length BOOLEAN,
                           prompt_version TEXT,
                           created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)
        await conn.execute(
            "ALTER TABLE request_log ALTER COLUMN trace_id DROP NOT NULL"
        )
        await conn.execute(
            "ALTER TABLE request_log ADD COLUMN IF NOT EXISTS is_error BOOLEAN NOT NULL DEFAULT FALSE"
        )
        await conn.execute(
            "ALTER TABLE request_log ADD COLUMN IF NOT EXISTS error_message TEXT"
        )
