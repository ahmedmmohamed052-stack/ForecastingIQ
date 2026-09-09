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
    "test_plan": {
        # TEMPORARY — for verifying the real Paymob checkout → webhook →
        # subscription-activation flow end-to-end without paying full
        # price. Remove this before real launch so customers never see it.
        "name": "Test Plan (remove before launch)",
        "price_usd": 0.50,
        "duration_days": 1,
        "most_popular": False,
        "tagline": "Internal — for testing the Paymob checkout flow only.",
        "max_forecast_months": 3,
        "max_historical_months": 12,
        "max_data_rows_per_month": 50_000,
        "max_trainings": 2,
        "max_forecast_points_per_month": 10_000,
        "max_models": 2,
        "product_level_forecasting": True,
        "advanced_forecasting": False,
        "features": ["Dashboard", "Data export", "2 saved trained models"],
    },
    "starter": {
        "name": "Starter",
        "price_usd": 29.0,
        "duration_days": 30,
        "most_popular": False,
        "tagline": "For a single store or product line getting started with forecasting.",
        "max_forecast_months": 3,
        "max_historical_months": 12,
        "max_data_rows_per_month": 50_000,
        "max_trainings": 10,
        "max_forecast_points_per_month": 10_000,
        "max_models": 2,
        "product_level_forecasting": True,
        "advanced_forecasting": False,
        "features": [
            "Up to 50K data rows/month",
            "10 training runs/month",
            "10K forecasted data points/month",
            "Up to 3-month forecast horizon",
            "Up to 12 months of historical data",
            "Up to 2 saved trained models",
            "Product-level forecasting",
            "Dashboard",
            "Data export",
        ],
    },
    "growth": {
        "name": "Growth",
        "price_usd": 79.0,
        "duration_days": 30,
        "most_popular": True,
        "tagline": "For growing businesses forecasting across many products or locations.",
        "max_forecast_months": 12,
        "max_historical_months": 36,
        "max_data_rows_per_month": 500_000,
        "max_trainings": 50,
        "max_forecast_points_per_month": 100_000,
        "max_models": 4,
        "product_level_forecasting": True,
        "advanced_forecasting": True,
        "features": [
            "Up to 500K data rows/month",
            "50 training runs/month",
            "100K forecasted data points/month",
            "Up to 12-month forecast horizon",
            "Up to 36 months of historical data",
            "Up to 4 saved trained models",
            "Advanced forecasting capabilities",
            "Data export",
        ],
    },
    "scale": {
        "name": "Scale",
        "price_usd": 149.0,
        "duration_days": 30,
        "most_popular": False,
        "tagline": "For high-volume operations that need long-horizon, large-scale forecasting.",
        "max_forecast_months": 24,
        "max_historical_months": 60,
        "max_data_rows_per_month": 2_000_000,
        "max_trainings": 200,
        "max_forecast_points_per_month": 500_000,
        "max_models": 6,
        "product_level_forecasting": True,
        "advanced_forecasting": True,
        "features": [
            "Up to 2M data rows/month",
            "200 training runs/month",
            "500K forecasted data points/month",
            "Up to 24-month forecast horizon",
            "60+ months of historical data",
            "Up to 6 saved trained models",
            "Advanced forecasting capabilities",
            "Data export",
        ],
    },
}

# Detailed, customer-friendly explanations for the Plan Details page. Keyed
# by the generic concept (not per-plan) — the frontend fills in each plan's
# actual numbers around this copy so the wording never has to be duplicated
# per plan and stays in sync automatically if a limit changes above.
PLAN_FEATURE_EXPLANATIONS = {
    "data_rows": {
        "title": "Data rows/month",
        "body": (
            "A data row is one line of your uploaded CSV — for example, one "
            "product's sales for one month. Your monthly limit is the total "
            "number of rows you can upload across all your training and "
            "forecast files combined."
        ),
    },
    "trainings": {
        "title": "Training runs/month",
        "body": (
            "A training run is when ForecastingIQ builds (or rebuilds) your "
            "forecasting model from your historical data. You'll typically "
            "use one whenever you add new historical data or want the model "
            "to learn from more recent trends."
        ),
    },
    "forecast_points": {
        "title": "Forecasted data points/month",
        "body": (
            "A forecasted data point is one predicted future value — for "
            "example, one product's predicted sales for one future month. "
            "Forecasting 500 products for 12 future months produces 6,000 "
            "forecasted data points."
        ),
    },
    "forecast_horizon": {
        "title": "Forecast horizon",
        "body": "How far into the future you can generate forecasts in a single run.",
    },
    "historical_data": {
        "title": "Historical data",
        "body": "How much past data ForecastingIQ can use to learn patterns and generate your forecasts.",
    },
    "product_level": {
        "title": "Product-level forecasting",
        "body": "Get a separate forecast for each individual product, store, or category instead of just one combined total.",
    },
    "max_models": {
        "title": "Saved trained models",
        "body": (
            "Every time you train, ForecastingIQ saves the resulting model so you can reuse it "
            "for future forecasts without retraining. Your plan caps how many trained models you "
            "can keep at once — delete an old one from the dashboard to free up a slot, or upgrade "
            "your plan for more."
        ),
    },
    "dashboard": {
        "title": "Dashboard",
        "body": "A visual dashboard where you can review your forecasts, track model accuracy, and manage your account.",
    },
    "data_export": {
        "title": "Data export",
        "body": "Download your forecasting results as a CSV file to use in spreadsheets, reports, or other business tools.",
    },
    "advanced_forecasting": {
        "title": "Advanced forecasting capabilities",
        "body": (
            "ForecastingIQ automatically tries several forecasting techniques on your "
            "data and additional input signals (like price or promotions) to pick "
            "the most accurate one for your business — no configuration needed."
        ),
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

    # ── Paymob (payment gateway — Egypt + Gulf/Middle East) ──────────────
    # All blank until you have real credentials from your Paymob dashboard.
    # See paymob.py — every function checks `settings.paymob_configured`
    # and returns a clear "not configured yet" error instead of crashing,
    # so the rest of the app works fine without these.
    #
    # Paymob operates (and settles) in several MENA markets, each under its
    # own currency and its own INTEGRATION_ID on your Paymob account — you
    # need one integration per currency you want to accept, even though
    # it's a single merchant account. Ask your Paymob account
    # manager/dashboard for the integration id for each currency you plan
    # to support; leave a country's PAYMOB_INTEGRATION_ID_<CODE> blank if
    # you're not accepting that currency yet, and it's simply hidden from
    # the country picker on the Plan Details page instead of erroring.
    PAYMOB_API_KEY: str = os.getenv("PAYMOB_API_KEY", "")
    PAYMOB_IFRAME_ID: str = os.getenv("PAYMOB_IFRAME_ID", "")
    PAYMOB_HMAC_SECRET: str = os.getenv("PAYMOB_HMAC_SECRET", "")

    # country code -> (currency, integration_id env var, USD conversion rate).
    # Rates are NOT a live lookup — set them from your bank/Paymob dashboard
    # and update manually whenever they meaningfully move, same as before.
    PAYMOB_COUNTRIES: dict = {
        "EG": {"label": "Egypt",         "currency": "EGP", "integration_id": os.getenv("PAYMOB_INTEGRATION_ID_EG", os.getenv("PAYMOB_INTEGRATION_ID", "")), "usd_rate": _get_float("USD_TO_EGP_RATE", 49.0)},
        "SA": {"label": "Saudi Arabia",  "currency": "SAR", "integration_id": os.getenv("PAYMOB_INTEGRATION_ID_SA", ""), "usd_rate": _get_float("USD_TO_SAR_RATE", 3.75)},
        "AE": {"label": "UAE",           "currency": "AED", "integration_id": os.getenv("PAYMOB_INTEGRATION_ID_AE", ""), "usd_rate": _get_float("USD_TO_AED_RATE", 3.67)},
        "OM": {"label": "Oman",          "currency": "OMR", "integration_id": os.getenv("PAYMOB_INTEGRATION_ID_OM", ""), "usd_rate": _get_float("USD_TO_OMR_RATE", 0.385)},
        "KW": {"label": "Kuwait",        "currency": "KWD", "integration_id": os.getenv("PAYMOB_INTEGRATION_ID_KW", ""), "usd_rate": _get_float("USD_TO_KWD_RATE", 0.307)},
        "QA": {"label": "Qatar",         "currency": "QAR", "integration_id": os.getenv("PAYMOB_INTEGRATION_ID_QA", ""), "usd_rate": _get_float("USD_TO_QAR_RATE", 3.64)},
        "BH": {"label": "Bahrain",       "currency": "BHD", "integration_id": os.getenv("PAYMOB_INTEGRATION_ID_BH", ""), "usd_rate": _get_float("USD_TO_BHD_RATE", 0.376)},
    }

    @property
    def paymob_configured(self) -> bool:
        # "Configured" overall just means Egypt works (the original,
        # always-expected setup) — individual Gulf countries light up
        # independently as you add their integration ids, checked per
        # country via paymob_countries_available below.
        return bool(self.PAYMOB_API_KEY and self.PAYMOB_IFRAME_ID and self.PAYMOB_COUNTRIES["EG"]["integration_id"])

    @property
    def paymob_countries_available(self) -> dict:
        """Only the countries that actually have an integration id set —
        what the Plan Details page's country picker should offer."""
        if not (self.PAYMOB_API_KEY and self.PAYMOB_IFRAME_ID):
            return {}
        return {code: info for code, info in self.PAYMOB_COUNTRIES.items() if info["integration_id"]}

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