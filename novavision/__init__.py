"NovaVision: emotion-guided images and paired emotion-recovery evaluation."

from novavision.pipeline import NovaVision, Result, build_pipeline

__version__ = "1.0.0"
__all__ = ["NovaVision", "Result", "build_pipeline", "run_experiment", "__version__"]


def __getattr__(name: str):
    if name == "run_experiment":
        from novavision.experiments.run import run_experiment

        return run_experiment
    raise AttributeError(f"module 'novavision' has no attribute {name!r}")
