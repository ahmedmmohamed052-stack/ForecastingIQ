"""
⚙️  Central configuration — everything environment-dependent lives here.

The whole point of this file: change ONE place (your .env) to switch the
app between "running on my laptop" and "running on a real server". Nothing
in main.py, Smart_Za3bola.py or paymob.py should ever hardcode a URL,
a password, or a "is this local?" check — they all read from `settings`.

Run modes (set ENVIRONMENT in .env):
  - "development" (default): permissive CORS, dev-password bypass allowed,
    verbose errors, fast/reduced training grid available via QUICK_TRAIN=true
  - "production": CORS locked to ALLOWED_ORIGINS, dev-password bypass is
    disabled outright (even if DEV_ACCESS_PASSWORD is set), full training grid
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# =============================================================================
# 💰  SUBSCRIPTION PLANS
# =============================================================================
# Single source of truth for pricing — used by /billing/plans, the Paymob
# checkout flow, quota enforcement, AND forecast-horizon enforcement.
#
# Plans are differentiated by how far ahead a user is allowed to forecast,
# not by how long their subscription lasts — every plan bills on the same
# 30-day cycle (usage quotas reset every 30 days regardless of plan). What
# you pay for is the forecast horizon:
#   - Forecast 3 months ahead  -> $30/mo
#   - Forecast 6 months ahead  -> $80/mo
#   - Forecast 12 months ahead -> $120/mo
# A plan's horizon is a ceiling, not a fixed value — e.g. the 6-month plan
# can request forecasts of 3, 6 months (anything up to and including its
# max_forecast_months), enforced in /forecast (see main.py).
PLANS = {
    "forecast_3mo": {
        "name": "3-Month Forecast",
        "price_usd": 30.0,
        "duration_days": 30,
        "max_forecast_months": 3,
        "max_trainings": 10,
        "max_forecasts": 100,
    },
    "forecast_6mo": {
        "name": "6-Month Forecast",
        "price_usd": 80.0,
        "duration_days": 30,
        "max_forecast_months": 6,
        "max_trainings": 25,
        "max_forecasts": 300,
    },
    "forecast_12mo": {
        "name": "12-Month Forecast",
        "price_usd": 120.0,
        "duration_days": 30,
        "max_forecast_months": 12,
        "max_trainings": 60,
        "max_forecasts": 800,
    },
}


class Settings:
    # ── Run mode ─────────────────────────────────────────────────────────
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").strip().lower()

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    # ── Server ───────────────────────────────────────────────────────────
    # HOST/PORT only matter when running via `python main.py`. If you deploy
    # behind something like Railway/Render/uvicorn-gunicorn, they set PORT
    # for you and this is respected automatically.
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Public URL of THIS backend once deployed (used only for informational
    # logging/docs — the frontend now auto-detects its API base from
    # window.location, so you no longer need to hand-edit HTML files).
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "")

    # ── CORS ─────────────────────────────────────────────────────────────
    # In development we stay permissive so you don't fight CORS while
    # building. In production, set ALLOWED_ORIGINS="https://yourapp.com,
    # https://www.yourapp.com" — wildcard "*" is refused when
    # ALLOW_CREDENTIALS is true anyway (browsers reject that combination).
    ALLOWED_ORIGINS: list[str] = _get_list("ALLOWED_ORIGINS", ["*"])

    # ── Firebase ─────────────────────────────────────────────────────────
    # Path to the service-account JSON key file (never commit this file —
    # it's in .gitignore already). Rotate it if it's ever shared/leaked.
    FIREBASE_CRED_PATH: str = os.getenv("FIREBASE_CRED_PATH", "firebase_key.json")

    # ── Gmail (forecast email delivery) ─────────────────────────────────
    GMAIL_SENDER: str = os.getenv("GMAIL_SENDER", "")
    GMAIL_APP_PASS: str = os.getenv("GMAIL_APP_PASS", "")

    @property
    def email_enabled(self) -> bool:
        return bool(self.GMAIL_SENDER and self.GMAIL_APP_PASS)

    # ── Training speed ───────────────────────────────────────────────────
    # The full grid search (2 models x 3 lag options x 3 rolling options,
    # with XGBoost's grid alone covering 1440 combinations x 5 CV folds)
    # is meant for a real production training job — it can take a long
    # time on a laptop. Set QUICK_TRAIN=true in .env for fast local
    # iteration; it shrinks the grid drastically. Leave it off (or unset)
    # in production for the full search.
    QUICK_TRAIN: bool = _get_bool("QUICK_TRAIN", default=False)

    # ── Developer trial access ──────────────────────────────────────────
    # Lets YOU (the developer) hit protected endpoints without a real
    # Firebase login, while you're building/demoing. Set DEV_ACCESS_PASSWORD
    # in .env, then send it as header `X-Dev-Password: <value>` (or query
    # param `?dev_key=<value>`) instead of an Authorization bearer token.
    #
    # SAFETY: this bypass is hard-disabled whenever ENVIRONMENT=production,
    # even if the password is still set in .env — so you can't accidentally
    # ship it live. Every use is also logged at WARNING level so it shows up
    # in your monitoring — see logging_setup.py.
    #
    # ⚠️ REMOVE / UNSET DEV_ACCESS_PASSWORD FROM YOUR PRODUCTION .env BEFORE
    # LAUNCH. The production hard-disable protects you from a misconfigured
    # ENVIRONMENT var, but an unset password is one less thing that can go
    # wrong. (Per your note — you'll tell us when it's time to remove it.)
    DEV_ACCESS_PASSWORD: str = os.getenv("DEV_ACCESS_PASSWORD", "")

    @property
    def dev_bypass_enabled(self) -> bool:
        return bool(self.DEV_ACCESS_PASSWORD) and not self.is_production

    # ── Paymob (payment gateway) ────────────────────────────────────────
    # All blank until you have real credentials from your Paymob dashboard.
    # See paymob.py — every function checks `settings.paymob_configured`
    # and returns a clear "not configured yet" error instead of crashing,
    # so the rest of the app works fine without these.
    PAYMOB_API_KEY: str = os.getenv("PAYMOB_API_KEY", "")
    PAYMOB_INTEGRATION_ID: str = os.getenv("PAYMOB_INTEGRATION_ID", "")
    PAYMOB_IFRAME_ID: str = os.getenv("PAYMOB_IFRAME_ID", "")
    PAYMOB_HMAC_SECRET: str = os.getenv("PAYMOB_HMAC_SECRET", "")

    # Plans are priced in USD above (matches how you think about pricing),
    # but Paymob (Egyptian merchant accounts) settles in EGP. Set your
    # actual conversion rate here — check your Paymob dashboard / bank for
    # what you're actually being credited, this is NOT a live exchange
    # rate lookup. Update it whenever the rate meaningfully moves.
    USD_TO_EGP_RATE: float = _get_float("USD_TO_EGP_RATE", 49.0)

    @property
    def paymob_configured(self) -> bool:
        return bool(
            self.PAYMOB_API_KEY
            and self.PAYMOB_INTEGRATION_ID
            and self.PAYMOB_IFRAME_ID
        )

    # ── Rate limiting ────────────────────────────────────────────────────
    # Protects /train and /forecast from being hammered (by a bot, a bug in
    # someone's client, or plain abuse) independent of subscription quotas.
    # This is a simple in-memory limiter — correct for a single server
    # process. If you later scale to multiple server instances/workers,
    # move this to a shared store (e.g. Redis) or it under-counts.
    RATE_LIMIT_TRAIN_PER_HOUR: int = _get_int("RATE_LIMIT_TRAIN_PER_HOUR", 5)
    RATE_LIMIT_FORECAST_PER_MINUTE: int = _get_int("RATE_LIMIT_FORECAST_PER_MINUTE", 10)

    # ── Background training queue ───────────────────────────────────────
    # /train no longer blocks the request while a multi-minute grid search
    # runs. It's queued and processed by a small worker pool instead — this
    # caps how many training jobs can run at once regardless of how many
    # requests come in, so the server can't be knocked over by concurrent
    # /train calls. See jobs.py.
    MAX_CONCURRENT_TRAINING_JOBS: int = _get_int("MAX_CONCURRENT_TRAINING_JOBS", 1)

    # ── Upload size limit ────────────────────────────────────────────────
    # Caps how large a CSV file /train and /forecast will accept, so a huge
    # (or malicious) upload can't blow up server memory/CPU before it ever
    # gets to validation. Enforced on actual bytes read, not the
    # Content-Length header (which a client can lie about) — see
    # read_upload_within_limit() in main.py.
    MAX_UPLOAD_SIZE_MB: int = _get_int("MAX_UPLOAD_SIZE_MB", 10)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # ── Monitoring / error tracking ──────────────────────────────────────
    # Leave blank to just log to console + a rotating local file
    # (see logging_setup.py). Set SENTRY_DSN (from sentry.io, free tier is
    # plenty to start) to also get real-time error alerts instead of having
    # to SSH in and read a log file after a customer complains.
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


settings = Settings()

