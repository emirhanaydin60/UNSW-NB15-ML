from __future__ import annotations

import logging
from pathlib import Path

from src.utils import ensure_dir


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for attr in ["run_id", "model", "outer_fold", "inner_fold", "gwo_iter", "wolf", "best_fitness"]:
            if not hasattr(record, attr):
                setattr(record, attr, "-")
        return True


def setup_logger(log_file: str | Path, level: str = "INFO") -> logging.Logger:
    ensure_dir(Path(log_file).parent)
    logger = logging.getLogger("unsw_nb15_benchmark")
    logger.setLevel(level.upper())
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [RUN=%(run_id)s] [MODEL=%(model)s] [OUTER_FOLD=%(outer_fold)s] " "[INNER_FOLD=%(inner_fold)s] [GWO_ITER=%(gwo_iter)s] [WOLF=%(wolf)s] " "[BEST_FITNESS=%(best_fitness)s] %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(ContextFilter())

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ContextFilter())

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def with_context(logger: logging.Logger, **ctx: object) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, extra=ctx)
