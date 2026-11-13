
#this is a signal factory, a way of fairly evaluating signals against one another

from . import signals  
from .benchmark import equal_weight_targets, eligible_universe, universe_breadth
from .catalogue import CATALOGUE, build_queue, catalogue_frame, cluster_of, entry
from .contracts import CASH, all_cash, normalise_targets, validate_panel, validate_schedule, validate_targets
from .costs import COST_MODES, per_side_rate, round_trip_bps
from .evaluate import active_returns, estimate_beta, evaluate_signal, run_portfolio
from .metrics import block_bootstrap_difference, information_ratio, summary
from .panel import daily_returns, pivot_closes, weekly_schedule
from .registry import available, build_targets, check_no_lookahead, defaults_for, register
from .screen import correlation_matrix, equal_risk_contribution, marginal_ir, portfolio_ir, screen_candidate, shrunk_covariance, stressed_correlation


__all__ = ["CASH", "CATALOGUE", "COST_MODES",
    "active_returns", "all_cash", "available", "block_bootstrap_difference", "build_queue", "build_targets",
    "catalogue_frame", "check_no_lookahead", "cluster_of", "correlation_matrix", "daily_returns",
    "defaults_for", "eligible_universe", "entry", "equal_risk_contribution", "equal_weight_targets",
    "estimate_beta", "evaluate_signal", "information_ratio", "marginal_ir", "normalise_targets",
    "per_side_rate", "pivot_closes", "portfolio_ir", "register", "round_trip_bps", "run_portfolio",
    "screen_candidate", "shrunk_covariance", "stressed_correlation", "summary", "universe_breadth",
    "validate_panel", "validate_schedule", "validate_targets", "weekly_schedule"]
