from src.evaluate_bias_in_nlg.eval_bias_in_labeled_generations import eval_regard_bias
from src.evaluate_bias_in_nlg.qualitative_eval import eval_qual_bias


def run_bias_evaluation(cfg):
    if cfg.run_mode.quant_eval:
        eval_regard_bias(cfg)
    if cfg.run_mode.qual_eval:
        eval_qual_bias(cfg)
