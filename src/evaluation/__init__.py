"""评测：主评测 / 鲁棒性测试 / 对比与消融实验。"""

from .evaluate import evaluate_model, format_report
from .robustness import run_robustness
from .ablation import run_ablation, ABLATION_PRESETS

__all__ = [
    "evaluate_model",
    "format_report",
    "run_robustness",
    "run_ablation",
    "ABLATION_PRESETS",
]
