"""core package — exposes public API for run_pipeline.py."""
from .preprocessor    import run_preprocessing
from .sentiment_engine import run_sentiment_inference
from .topic_engine    import run_topic_severity
from .severity_analyzer import run_severity_mapping
from .crawler import run_standalone_crawler

__all__ = [
    "run_preprocessing",
    "run_sentiment_inference",
    "run_topic_severity",
    "run_severity_mapping",
    "run_standalone_crawler",
]

