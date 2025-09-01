import math, random, numpy as np, pandas as pd
from copy import deepcopy
from tqdm import tqdm
import inspect
from typing import Optional, Callable, Dict, Any, Tuple
# If these imports differ in your tree, adjust the path accordingly:
from strategy_tester import StrategyTester  # your tester module

# ---- objective blending + constraints ----
def score_from_metrics(m: dict) -> float:
    # Hard constraints
    if (m.get('total_trades', 0) < 30 or
        m.get('profit_factor', 0) < 1.4 or
        m.get('max_drawdown', 1) > 0.12 or
        not (5 <= m.get('exposure_percentage', 0) <= 85)):
        return 1e-6  # heavy penalty

    calmar = float(m.get('calmar_ratio', 0) or 0)
    sharpe = float(m.get('sharpe_ratio', 0) or 0)
    pf     = float(m.get('profit_factor', 0) or 1.01)

    # Composite score
    return 0.6*calmar + 0.3*sharpe + 0.1*math.log(max(pf, 1.01))

# ---- run a single backtest with safe kwargs ----
def run_bt(params: dict, data: pd.DataFrame):
    tester = StrategyTester(initial_capital=10_000)

    # Keep your strategy defaults (TPs etc.) unless overridden
    base = dict(
        sensitivity=18,
        leverage=1.0,
        fixed_stop=False,
        sl_percent=0.0,
        break_even_after_tp=1,
        use_take_profits=True,
        tp1_pct=1.0, tp1_close=40.0,
        tp2_pct=2.0, tp2_close=30.0,
        tp3_pct=3.0, tp3_close=20.0,
        tp4_pct=4.0, tp4_close=10.0,
        risk_per_trade=0.02,
    )
    base.update(params)

    # Some versions of your class may not accept 'timeframe' kwarg; fall back if needed
    try:
        res = tester.run_backtest('FibonacciChannelStrategy', data, verbose=False, **base)
    except TypeError:
        base2 = {k: v for k, v in base.items() if k != 'timeframe'}
        res = tester.run_backtest('FibonacciChannelStrategy', data, verbose=False, **base2)

    return res['metrics']

# ---- sample a parameter set from the search space ----
def sample_params(rng: random.Random) -> dict:
    return {
        # keep timeframe if your ctor accepts it; remove if it raises TypeError
        "timeframe": "15m",

        "sensitivity": rng.randint(5, 50),

        # fib stop adjustment (+/- 0.50% around fib_786/fib_236)
        "sl_percent": 0, #round(rng.uniform(-0.50, 0.50), 2),

        "fixed_stop": False,  # you said you want fib-based stops

        "break_even_after_tp": rng.choice([1]),

        # log-uniform between 0.3% and 4.0%
        "risk_per_trade": 2 #round(10 ** rng.uniform(math.log10(0.01), math.log10(0.04)), 4),
    }

def optimize(train_df: pd.DataFrame,
             val_df: pd.DataFrame | None = None,
             n_trials: int = 200,
             seed: int = 42,
             topk: int = 25,
             show_progress: bool = True):
    rng = random.Random(seed)
    leaderboard = []

    # --- Random search on TRAIN ---
    train_iter = range(1, n_trials + 1)
    if show_progress:
        train_iter = tqdm(train_iter, desc="Optimizing (train)", unit="trial")

    best_train_score = -float("inf")
    for t in train_iter:
        params = sample_params(rng)

        metrics_tr = run_bt(params, train_df)
        score_tr   = score_from_metrics(metrics_tr)

        leaderboard.append({
            "trial": t,
            "params": deepcopy(params),
            "score_train": score_tr,
            **{f"train_{k}": v for k, v in metrics_tr.items()}
        })

        if score_tr > best_train_score:
            best_train_score = score_tr

        if show_progress:
            train_iter.set_postfix({
                "best_train": f"{best_train_score:.3f}",
                "last_pf": f"{metrics_tr.get('profit_factor', 0):.2f}",
                "last_mdd": f"{metrics_tr.get('max_drawdown', 0):.2%}"
            })

    # sort by train score
    leaderboard.sort(key=lambda x: x["score_train"], reverse=True)

    # --- Optional: re-rank top K on VALIDATION ---
    if val_df is not None and topk > 0:
        topK = leaderboard[:topk]

        val_iter = topK
        if show_progress:
            val_iter = tqdm(topK, desc=f"Validating top-{topk}", unit="trial")

        best_val_score = -float("inf")
        for item in val_iter:
            params = item["params"]
            metrics_val = run_bt(params, val_df)
            score_val   = score_from_metrics(metrics_val)

            item["score_val"] = score_val
            item.update({f"val_{k}": v for k, v in metrics_val.items()})

            if score_val > best_val_score:
                best_val_score = score_val

            if show_progress:
                val_iter.set_postfix({
                    "best_val": f"{best_val_score:.3f}",
                    "pf": f"{metrics_val.get('profit_factor', 0):.2f}",
                    "mdd": f"{metrics_val.get('max_drawdown', 0):.2%}"
                })

        # Combine scores (favor out-of-sample)
        for item in topK:
            st = item["score_train"]
            sv = item.get("score_val", 0)
            item["score_combined"] = 0.3 * st + 0.7 * sv

        topK.sort(key=lambda x: x["score_combined"], reverse=True)
        best = topK[0]
    else:
        best = leaderboard[0]

    return best, leaderboard

def _pick_runner(runner: Optional[Callable]):
    """Choose the backtest function (defaults to global run_bt)."""
    if runner is not None:
        return runner
    try:
        return run_bt  # use your existing helper
    except NameError:
        raise RuntimeError("No backtest runner found. Pass runner=... or define run_bt(...) in this module.")

def _pick_scorer(scoring_fn: Optional[Callable[[Dict[str, Any]], float]]):
    """Choose the scoring function (defaults to your score_from_metrics if present)."""
    if scoring_fn is not None:
        return scoring_fn
    try:
        return score_from_metrics  # your project-level scorer, if present
    except NameError:
        # sensible default: return - 0.5*MDD + 0.1*(PF-1) + 0.05*(WR-0.5)
        def _default(m):
            ret = m.get("total_return", 0.0)
            mdd = m.get("max_drawdown", 0.0)
            pf  = m.get("profit_factor", 0.0)
            wr  = m.get("win_rate", 0.0)
            return (ret - 0.5*mdd) + 0.1*(pf - 1.0) + 0.05*(wr - 0.5)
        return _default

def _extract_metrics(bt_result: Any) -> Dict[str, Any]:
    """Accept either {'metrics': {...}} or a metrics dict directly."""
    if isinstance(bt_result, dict) and "metrics" in bt_result and isinstance(bt_result["metrics"], dict):
        return bt_result["metrics"]
    if isinstance(bt_result, dict):
        return bt_result
    raise ValueError("Backtest result not understood; expected dict or {'metrics': {...}}.")

def _call_run_bt(runner: Callable, strategy_name: str, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the project's run_bt with flexible signatures:
      1) runner(strategy_name, df, **params)
      2) runner(params, df)
      3) runner(df, **params)
    Returns metrics dict.
    """
    sig = inspect.signature(runner)
    kwargs = dict(**params)
    try:
        # try (name, df, **params)
        if len(sig.parameters) >= 2:
            out = runner(strategy_name, df, **kwargs)
            return _extract_metrics(out)
    except TypeError:
        pass

    try:
        # try (params, df)
        out = runner(params, df)
        return _extract_metrics(out)
    except TypeError:
        pass

    # fallback (df, **params)
    out = runner(df, **kwargs)
    return _extract_metrics(out)

def optimize_sensitivity(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    base_params: Optional[Dict[str, Any]] = None,
    strategy_name: str = "FibonacciChannelStrategy",
    sens_min: int = 4,
    sens_max: int = 60,
    coarse_step: int = 3,
    fine_radius: int = 4,
    topk_val: int = 10,
    scoring_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    runner: Optional[Callable] = None,
    show_progress: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Systematic coarse→fine grid search over 'sensitivity' ONLY.
    All other params stay as in base_params.

    Returns:
        best: dict of the best row (params + train/val metrics/scores)
        leaderboard: DataFrame with all tried sensitivities and metrics
    """
    if base_params is None:
        base_params = {}

    runner = _pick_runner(runner)
    scorer = _pick_scorer(scoring_fn)

    # ----------------------
    # Phase 1: coarse sweep
    # ----------------------
    results = []
    coarse_grid = list(range(sens_min, sens_max + 1, max(1, int(coarse_step))))
    itr = tqdm(coarse_grid, desc="Sensitivity (coarse)", unit="sens", disable=not show_progress)

    best_s = None
    best_score_tr = -float("inf")

    for s in itr:
        params = deepcopy(base_params)
        params["sensitivity"] = int(s)

        m_tr = _call_run_bt(runner, strategy_name, train_df, params)
        score_tr = scorer(m_tr)

        row = {"sensitivity": s, "score_train": score_tr}
        row.update({f"train_{k}": v for k, v in m_tr.items()})
        results.append(row)

        if score_tr > best_score_tr:
            best_score_tr = score_tr
            best_s = s

        itr.set_postfix({
            "best_sens": best_s,
            "best_score": f"{best_score_tr:.3f}",
            "PF": f"{m_tr.get('profit_factor', 0):.2f}",
            "MDD": f"{m_tr.get('max_drawdown', 0):.2%}",
        })

    # ----------------------
    # Phase 2: fine sweep
    # ----------------------
    lo = max(sens_min, best_s - fine_radius)
    hi = min(sens_max, best_s + fine_radius)
    fine_grid = [x for x in range(lo, hi + 1) if x not in coarse_grid]

    if fine_grid:
        itr = tqdm(fine_grid, desc="Sensitivity (fine)", unit="sens", disable=not show_progress)
        for s in itr:
            params = deepcopy(base_params)
            params["sensitivity"] = int(s)

            m_tr = _call_run_bt(runner, strategy_name, train_df, params)
            score_tr = scorer(m_tr)

            row = {"sensitivity": s, "score_train": score_tr}
            row.update({f"train_{k}": v for k, v in m_tr.items()})
            results.append(row)

            if score_tr > best_score_tr:
                best_score_tr = score_tr
                best_s = s

            itr.set_postfix({
                "best_sens": best_s,
                "best_score": f"{best_score_tr:.3f}",
                "PF": f"{m_tr.get('profit_factor', 0):.2f}",
                "MDD": f"{m_tr.get('max_drawdown', 0):.2%}",
            })

    # Build leaderboard
    table = pd.DataFrame(results).sort_values("score_train", ascending=False).reset_index(drop=True)
    best = table.iloc[0].to_dict()

    # Optional: validate top-K on holdout
    if val_df is not None and topk_val > 0:
        top = table.head(min(topk_val, len(table))).copy()
        val_rows = []
        itr = tqdm(top["sensitivity"], desc=f"Validating top-{len(top)}", unit="sens", disable=not show_progress)
        for s in itr:
            params = deepcopy(base_params); params["sensitivity"] = int(s)
            m_val = _call_run_bt(runner, strategy_name, val_df, params)
            score_val = scorer(m_val)
            r = {"sensitivity": s, "score_val": score_val}
            r.update({f"val_{k}": v for k, v in m_val.items()})
            val_rows.append(r)
            itr.set_postfix({"score_val": f"{score_val:.3f}"})

        val_df_scores = pd.DataFrame(val_rows)
        table = table.merge(val_df_scores, on="sensitivity", how="left")
        # Combine scores (favor validation)
        table["score_combined"] = 0.3*table["score_train"] + 0.7*table["score_val"].fillna(-np.inf)
        table = table.sort_values(["score_combined", "score_train"], ascending=False).reset_index(drop=True)
        best = table.iloc[0].to_dict()

    return best, table

# ---------- Example usage ----------
# best, board = optimize(train, val_df=valid, n_trials=300, seed=123)
# print("Best params:", best["params"])
# print("Train PF/Sharpe/MDD:",
#       best.get("train_profit_factor"), best.get("train_sharpe_ratio"), best.get("train_max_drawdown"))
# if "score_val" in best:
#     print("Val  PF/Sharpe/MDD:",
#           best.get("val_profit_factor"), best.get("val_sharpe_ratio"), best.get("val_max_drawdown"))
#
# # Turn leaderboard into a DataFrame to inspect the frontier
# lb_df = pd.DataFrame(board)
# lb_df.head()
