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


async def init_eval_tables():

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                           id SERIAL PRIMARY KEY,
                           run_id UUID NOT NULL UNIQUE,
                           dataset_version TEXT NOT NULL,
                           generator_model TEXT NOT NULL,
                           judge_model TEXT NOT NULL,
                           total_items INT NOT NULL,
                           heuristic_pass_rate REAL,
                           retrieval_recall_at_5 REAL,
                           retrieval_mrr REAL,
                           retrieval_ndcg_at_5 REAL,
                           judge_pass_rate REAL,
                           started_at TIMESTAMPTZ NOT NULL,
                           finished_at TIMESTAMPTZ,
                           created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                           )
            """)
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_results (
                           id SERIAL PRIMARY KEY,
                           run_id UUID NOT NULL REFERENCES eval_runs(run_id),
                           item_id TEXT NOT NULL,
                           category TEXT NOT NULL,
                           heuristic_cites_source BOOLEAN,
                           heuristic_grounded_refusal BOOLEAN,
                           retrieval_rank INT,
                           judge_faithful BOOLEAN,
                           judge_relevant BOOLEAN,
                           judge_complete BOOLEAN,
                           latency_ms INT,
                           raw_answer TEXT,
                           error TEXT,
                           created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                           )
            """)


async def migrate_request_log_add_answer():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE request_log ADD COLUMN IF NOT EXISTS answer TEXT"
        )


async def init_feedback_table():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                            id SERIAL PRIMARY KEY,
                            trace_id TEXT NOT NULL,
                            query TEXT NOT NULL,
                           answer TEXT NOT NULL,
                           rating TEXT NOT NULL, -- 'up' | 'down'
                           reason TEXT,
                           signal_type TEXT NOT NULL, -- 'explicit' | 'implicit_retry'
                           promoted BOOLEAN NOT NULL DEFAULT false,
                           created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                           )
            """)
