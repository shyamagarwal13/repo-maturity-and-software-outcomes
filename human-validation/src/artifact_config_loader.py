"""IMPORT-ONLY STUB for the shareable validation package.

The real module lives in stanford-swepr/ai-artifacts-collector. It is imported
at the top of maturity_scorer.py but none of its functions are reached by
notebook 15 (which only uses CATEGORY_TO_LEVEL, FileClassification, and
aggregate_repo_maturity).
"""

_MSG = ("stubbed in the shareable validation package — requires the "
        "ai-artifacts-collector repo (see README.md)")


def load_json_configs(*args, **kwargs):
    raise NotImplementedError(f"load_json_configs is {_MSG}")


def load_shared_config(*args, **kwargs):
    raise NotImplementedError(f"load_shared_config is {_MSG}")
