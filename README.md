# Sales Forecasting SaaS 📈

This project builds an advanced time series forecasting system to predict future product sales using historical e-commerce data.
It compares multiple models, performs extensive feature engineering, and selects the best configuration using a custom composite score — served through a FastAPI backend and a browser dashboard.

---

## 🚀 Quick start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your environment
cp .env.example .env
# then edit .env — at minimum, set FIREBASE_CRED_PATH to point at your
# Firebase service-account JSON key (Firebase Console -> Project Settings
# -> Service Accounts -> Generate new private key), and set a
# DEV_ACCESS_PASSWORD so you can try the app without wiring up a real login.

# 3. Run it
python main.py
```

Then open:
- **App:** http://127.0.0.1:8000/
- **Dashboard:** http://127.0.0.1:8000/dashboard
- **API docs (Swagger):** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

### Trying it as a developer (no real login needed)
Set `DEV_ACCESS_PASSWORD` in `.env`, then call any protected endpoint with
either:
- header `X-Dev-Password: <your password>`
- query string `?dev_key=<your password>`

instead of a Firebase bearer token. This bypass is **automatically disabled**
whenever `ENVIRONMENT=production`, so it can't accidentally ship live.
Every use is also logged at `WARNING` level. It also skips subscription/quota
checks entirely, since it's meant for you, not a paying customer.

> ⚠️ **Before launching publicly:** remove/blank `DEV_ACCESS_PASSWORD` from
> your production `.env` entirely. The production hard-disable protects
> against a misconfigured `ENVIRONMENT` var, but an unset password removes
> the risk altogether.

### Faster local training
Grid search by default is tuned for production-quality models and can take a
while. Set `QUICK_TRAIN=true` in `.env` (already the default in
`.env.example`) to shrink the search space for fast local iteration.

---

## ☁️ Going from local to a real server

Nothing in the code needs to change. Set these in your server's environment
instead of `.env`:

```
ENVIRONMENT=production
ALLOWED_ORIGINS=https://yourapp.com,https://www.yourapp.com
PORT=<provided by your host, e.g. Railway/Render sets this automatically>
```

Then start it with a real ASGI server instead of `python main.py`:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The two frontend pages (`Sales-Forecast-app.html`, `dashboard.html`) detect
their API base from the page's own origin automatically — no URLs to
hand-edit when you move between local and deployed.

---

## 💰 Subscription plans

Three plans, priced by how far ahead you're allowed to forecast (not by
how long the subscription lasts — every plan bills on the same 30-day
cycle):

| Plan | Price | Forecast horizon | Trainings/mo | Forecasts/mo |
|---|---|---|---|---|
| `forecast_3mo`  | $30/mo  | up to 3 months ahead  | 10 | 100 |
| `forecast_6mo`  | $80/mo  | up to 6 months ahead  | 25 | 300 |
| `forecast_12mo` | $120/mo | up to 12 months ahead | 60 | 800 |

Change prices/limits in `config.py` (`PLANS` dict) — everything else
(checkout, quota enforcement, horizon enforcement) reads from there.

- `GET /billing/plans` — public, lists the plans above
- `POST /billing/subscribe` — starts a Paymob checkout for a plan
- `GET /billing/status` — current user's plan + usage this cycle
- Requesting a forecast horizon beyond your plan's limit (e.g. asking a
  $30 plan for a 12-month forecast) returns `402` with a clear message —
  checked *before* it counts against your usage, so a rejected request
  never burns a quota slot
- No active subscription at all -> `/train` and `/forecast` return `402`

## 💳 Payments (Paymob)

Paymob integration is fully wired in `paymob.py`, `billing.py` and the
`/billing/*` + `/payment/webhook` endpoints, but stays **inactive** until
you add real credentials to `.env`:

```
PAYMOB_API_KEY=
PAYMOB_INTEGRATION_ID=
PAYMOB_IFRAME_ID=
PAYMOB_HMAC_SECRET=
USD_TO_EGP_RATE=49.0   # plans are priced in USD; Paymob settles in EGP — set your real rate
```

Get these from your Paymob dashboard (https://accept.paymob.com/portal2/en/login).
Until they're set, `/billing/subscribe` returns a clear `503` instead of
crashing — everything else in the app works normally without payments.

Flow once configured: `/billing/subscribe` creates a Paymob order and
returns an iframe URL -> user pays -> Paymob calls `/payment/webhook` ->
webhook verifies the HMAC signature -> subscription activates
automatically in Firestore.

---

## 🚦 Rate limiting & background training

`/train` no longer blocks the request while a multi-minute grid search
runs — it's queued and processed by a small worker pool
(`MAX_CONCURRENT_TRAINING_JOBS` in `.env`, default 1), so the server stays
responsive no matter how many training requests come in at once:

1. `POST /train` validates the upload and returns a `job_id` immediately
2. `GET /train/status/{job_id}` — poll until `status` is `done` or `failed`
3. Once `done`, the model is already saved — call `/forecast` normally

On top of subscription quotas, both endpoints are also rate-limited
independent of plan (`RATE_LIMIT_TRAIN_PER_HOUR`, `RATE_LIMIT_FORECAST_PER_MINUTE`
in `.env`) — this stops a bot or a buggy client from hammering the server
even if the user technically has quota left. This in-memory limiter is
correct for a single server process; if you scale to multiple instances,
move it to a shared store (e.g. Redis).

---

## 📋 Monitoring

Logs go to console + a rotating file (`logs/forecastiq.log`) instead of
scattered `print()` calls. For real-time error alerts instead of having to
SSH in and grep a log file, set `SENTRY_DSN` in `.env` (free tier at
sentry.io is plenty to start) and `pip install sentry-sdk`.

---

## 🔐 Firestore security rules

`firestore.rules` locks Firestore down to **backend-only access** — your
frontend never talks to Firestore directly (only Firebase Auth), so all
client-side reads/writes are denied outright. Apply it via Firebase
Console -> Firestore Database -> Rules -> paste -> Publish (or
`firebase deploy --only firestore:rules` via the CLI). See the comments
in that file for why this is the right fix, not just *a* fix.

---

## 🔐 Security notes
- `firebase_key.json` and `.env` are gitignored — never commit either.
- If a Firebase service-account key or `.env` file is ever shared, pasted
  into a chat, or committed by accident, rotate/regenerate it immediately
  from the Firebase Console.
- In production, set `ALLOWED_ORIGINS` to your real domain(s) — a wildcard
  origin with credentials enabled is rejected by browsers anyway, so this
  also fixes a previously-broken CORS configuration.

---

## 🔍 Problem Statement
Accurate sales forecasting is critical for inventory planning, pricing strategies, and promotional decisions.  
This project aims to predict future product purchases by learning seasonal patterns, lagged behavior, and customer conversion dynamics.

---

## 🧠 Key Features
- Time series–aware modeling using lag and rolling window features
- Multiple models comparison (Ridge Regression vs XGBoost)
- Hyperparameter tuning with TimeSeriesSplit
- Baseline comparison using naïve lag prediction
- Multi-step future forecasting
- Product-level performance analysis and visualization

---

## ⚙️ Feature Engineering
- Lag features (configurable number of past periods)
- Rolling mean and rolling standard deviation
- Conversion rate
- Cart drop rate
- One-hot encoding for product IDs
- Iterative imputation for missing values
- Standard scaling for numerical features

---

## 🧪 Models Used
- **Ridge Regression**
- **XGBoost Regressor**

Each model is evaluated using RMSE and compared against a baseline forecast.

---

## 🏆 Model Selection Strategy
A custom composite score is used to select the best model configuration:

