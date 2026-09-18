"""TimesFM (Google Research) per ball position — a foundation model, nothing fitted.

Every other predictor here is fitted on this history: Prophet, the statsforecast
trio and XGBoost all learn their parameters from the draws they are given.
TimesFM does not. It is a pretrained transformer that takes a context window of
values and emits a forecast **zero-shot**, so what it brings to this project is
not a better fit but a different question: *does a model trained on a very large
corpus of real time series see anything in Baloto that a model fitted on Baloto
alone does not?*

The answer the backtest gives is no, and that is the interesting part. This
repository exists around the premise that Baloto draws are i.i.d. uniform by
design ([docs/domain-and-premise.md](../../docs/domain-and-premise.md)), so a
foundation model has nothing to find either — and a foundation model failing the
same chance test as a 3-line frequency baseline is a stronger statement than the
baseline failing alone. It is scored through exactly the same `_score_window` and
the same one-sided z-test as everything else, so the claim is measured rather
than asserted.

**This project runs locally, which is what makes the defaults here the strong
ones.** An earlier version of this module was shaped around hosting a public
dashboard: it pinned the Apache-2.0 2.5 checkpoint because a public URL is
production, and it clipped the context short. Neither constraint applies to a
checkout on your own machine, so the defaults are now the 3.0 checkpoint, the
full history as context, and the GPU when there is one. See
[docs/local-setup.md](../../docs/local-setup.md).

Two structural details worth knowing:

**One call forecasts every position at once.** TimesFM takes a list of arrays as
a batch, so six positions are six rows of one forward pass rather than six passes
— the same reason `statsforecast_model.fit_predict_all` does all positions in one
call and this project tells you not to loop over positions for it.

**The forecaster is injectable.** `forecast_positions(..., forecaster=...)` takes
any object with `.forecast(horizon, inputs)`. That is what lets the tests pin
this module's real invariants — clipping, position semantics, batching, context
truncation — without downloading a checkpoint, and it is the seam the two
checkpoint families plug into: 2.5 and 3.0 ship genuinely different APIs, and
`_Timesfm3Adapter` below is what makes that difference stop at this module's
edge.
"""

import importlib.util

import numpy as np

from lottery.models.common import clip_to_range

# TimesFM 3.0 — the newest and strongest checkpoint, and the default because this
# project runs locally.
#
# **Licence, which is not the same answer as the rest of this project.** Upstream
# distributes TimesFM weights up to 2.5 under Apache-2.0, and 3.0's weights under
# a separate `timesfm-non-commercial-license-v1.0` restricted to non-commercial,
# non-production use. Local research on your own machine is exactly that, so 3.0
# is the right default here — but it is a real restriction, not a formality: if
# this ever goes back onto a public URL or into anything commercial, switch to
# CHECKPOINT_APACHE below. The `timesfm` package itself is Apache-2.0; only the
# weights differ, and nothing in this repository redistributes them.
CHECKPOINT = "google/timesfm-3.0-pytorch"

# The Apache-2.0 alternative, unrestricted in use. Slightly older and weaker.
CHECKPOINT_APACHE = "google/timesfm-2.5-200m-pytorch"

# Context in draws. A full 2010-2026 Baloto export is ~1035 draws, so 2048 hands
# the model the entire history with room to spare and truncation never bites.
# TimesFM 3.0 accepts up to 16k. The 2.5 checkpoint has a lower internal limit
# and its loader raises a message naming it — see `_load_2p5`.
MAX_CONTEXT = 2048

# Compiled horizon. One draw for the dashboard; a frozen holdout forecasts the
# whole remaining stretch, which for a long cutoff is well under this.
MAX_HORIZON = 128

# Below this many draws there is no context worth calling a foundation model on,
# and the backtest skips the window rather than scoring a guess made from noise.
# It is the same role `min_history_required` plays for XGBoost.
MIN_CONTEXT = 32


def is_available():
    """True when `timesfm` and `torch` can be imported, **without importing them**.

    `find_spec` only resolves the module on the path. Importing torch to find out
    whether torch is installed would cost seconds and gigabytes of address space
    on every dashboard rerun, which is the opposite of what the caller is asking.
    """
    return all(importlib.util.find_spec(name) is not None for name in ("timesfm", "torch"))


def best_device():
    """`"cuda"` when this machine has a GPU, else `"cpu"`. Imports torch, so call late.

    Running locally is what makes this worth doing at all: a hosted free tier has
    no GPU to find. Kept separate from `load_forecaster` so a caller can report
    what will be used before paying for the load.
    """
    import torch  # noqa: PLC0415 — deliberately lazy; see the module docstring

    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon. `mps` is absent on older torch builds, hence getattr.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class CheckpointUnavailableError(RuntimeError):
    """The package is installed but the weights could not be fetched.

    Its own type because it is the *expected* failure, not a bug: the checkpoint
    lives on HuggingFace and the first call is a network download. Offline, behind
    a proxy, or on a host that blocks huggingface.co, what the library raises is a
    bare `ProxyError: 403 Forbidden` — measured, in this project's own sandbox —
    which tells a dashboard reader nothing at all. Callers catch this instead and
    can say what actually went wrong.
    """


class _Timesfm3Adapter:
    """Gives the 3.0 forecaster the same `.forecast(horizon, inputs)` as 2.5.

    The two checkpoint families do not share an API: 2.5 is compiled with a
    `ForecastConfig` and exposes `.forecast(horizon, inputs) -> (point, quantiles)`,
    while 3.0 is constructed with a device and exposes
    `.predict_batch(contexts, horizon, ...) -> Iterator[ForecastOutput]`. Rather
    than let that difference leak into `forecast_matrix` — and from there into the
    backtest, the dashboard and every test — it stops here, in six lines.
    """

    def __init__(self, forecaster):
        self._forecaster = forecaster

    def forecast(self, horizon, inputs):
        outputs = list(self._forecaster.predict_batch(contexts=list(inputs), horizon=horizon))
        # ForecastOutput.forecast is (horizon,) for a univariate context. Stacking
        # gives the (batch, horizon) array the 2.5 API returns directly.
        point = np.stack([np.asarray(out.forecast).reshape(-1)[:horizon] for out in outputs])
        return point, None


_FORECASTER = {}


def _load_2p5(timesfm, checkpoint, max_context, max_horizon):
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint)
    try:
        model.compile(timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            # Each ball position lives on a fixed small integer range, not a
            # trending scale, but normalising is still what the checkpoint was
            # trained to expect and it costs nothing here.
            normalize_inputs=True,
        ))
    except ValueError as exc:
        # The library checks `max_context + max_horizon` against the checkpoint's
        # own limit and names it in the message. Re-raised with the knob to turn.
        raise CheckpointUnavailableError(
            f"The {checkpoint!r} checkpoint rejected a context of {max_context} with a horizon "
            f"of {max_horizon}: {exc} Lower timesfm_model.MAX_CONTEXT, or use the default 3.0 "
            f"checkpoint, which accepts far longer contexts."
        ) from exc
    return model


def load_forecaster(checkpoint=CHECKPOINT, max_context=MAX_CONTEXT, max_horizon=MAX_HORIZON,
                    device=None):
    """Load the checkpoint once per process, memoised by its arguments.

    First call downloads the weights from HuggingFace (a few hundred MB) into the
    usual HF cache; later calls in the same process are free. This is a plain dict
    rather than `functools.lru_cache` so a caller can inspect or clear it, and so
    the failure of one checkpoint never poisons the memo for another.

    `device` defaults to the best this machine has — the GPU when there is one,
    which is the main thing running locally buys you over a hosted free tier. It
    is honoured on the 3.0 checkpoints, whose loader takes it; 2.5 exposes no such
    knob and runs where it runs.
    """
    resolved_device = device or (best_device() if is_available() else "cpu")
    key = (checkpoint, max_context, max_horizon, resolved_device)
    if key not in _FORECASTER:
        import timesfm  # noqa: PLC0415 — deliberately lazy; see the module docstring

        try:
            if "3.0" in checkpoint or "3p0" in checkpoint:
                model = _Timesfm3Adapter(
                    timesfm.TimesFM3Forecaster.from_pretrained(checkpoint,
                                                               device=resolved_device))
            else:
                model = _load_2p5(timesfm, checkpoint, max_context, max_horizon)
        except CheckpointUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 — the library raises whatever the transport raised
            raise CheckpointUnavailableError(
                f"Could not load the TimesFM checkpoint {checkpoint!r}. The weights are downloaded "
                f"from HuggingFace on first use and cached under ~/.cache/huggingface, so this "
                f"usually means no network, a proxy, or a host that blocks huggingface.co. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

        _FORECASTER[key] = model
    return _FORECASTER[key]


def _contexts(position_series, upto=None, max_context=MAX_CONTEXT):
    """The last `max_context` observed values per position, in position order.

    `upto` truncates exactly like every other predictor in this project: rows
    `[:upto]`, so the draw at index `upto` — the one being predicted — is never
    in the context. Getting this wrong is the temporal leak that
    `xgboost_model` was written to fix, and it would be invisible in the output.
    """
    positions = sorted(position_series)
    arrays = []
    for position in positions:
        frame = position_series[position]
        values = (frame if upto is None else frame.iloc[:upto])["y"].to_numpy(dtype=np.float32)
        arrays.append(values[-max_context:])
    return positions, arrays


def forecast_matrix(position_series, n_columns, horizon=1, upto=None, forecaster=None,
                    checkpoint=CHECKPOINT, max_context=MAX_CONTEXT):
    """`{step: {position: number}}` for `horizon` steps ahead, every position in one pass.

    The core both public helpers below are built from. Returns None when the
    context is too short, which is the contract `lottery/backtest.py` expects for
    a window a model cannot predict — never a fallback guess, which would score
    free hits into the chance test.
    """
    positions, arrays = _contexts(position_series, upto=upto, max_context=max_context)
    if not arrays or min(len(a) for a in arrays) < MIN_CONTEXT:
        return None

    model = forecaster if forecaster is not None else load_forecaster(
        checkpoint=checkpoint, max_context=max_context)
    point_forecast, _quantiles = model.forecast(horizon=horizon, inputs=arrays)
    point_forecast = np.asarray(point_forecast)

    by_step = {}
    for row, position in enumerate(positions):
        for step in range(horizon):
            # Clipping is not cosmetic: the model knows nothing about 1-43 or
            # 1-16 and will happily return 44.7 or a negative number. Every model
            # in this project goes through clip_to_range for the same reason.
            by_step.setdefault(step, {})[position] = clip_to_range(
                float(point_forecast[row, step]), position, n_columns)
    return by_step


def forecast_positions(position_series, n_columns, upto=None, forecaster=None,
                       checkpoint=CHECKPOINT, max_context=MAX_CONTEXT):
    """One draw ahead: `{position: number}`, or None when the context is too short."""
    by_step = forecast_matrix(position_series, n_columns, horizon=1, upto=upto,
                              forecaster=forecaster, checkpoint=checkpoint,
                              max_context=max_context)
    return None if by_step is None else by_step[0]


def forecast_horizon_positions(position_series, n_columns, start, horizon, forecaster=None,
                               checkpoint=CHECKPOINT, max_context=MAX_CONTEXT):
    """Frozen-mode forecast: everything from `start`, from one pass over `[:start]`.

    Nothing after `start` is ever read, so this is the literal "what did it say in
    July about August?" experiment. Unlike `xgboost_model.forecast_horizon` there
    is no recursive feedback of the model's own output — TimesFM emits the whole
    horizon in one shot — so this cannot flatten to a fixed point the way the lag
    models do.
    """
    return forecast_matrix(position_series, n_columns, horizon=horizon, upto=start,
                           forecaster=forecaster, checkpoint=checkpoint, max_context=max_context)
