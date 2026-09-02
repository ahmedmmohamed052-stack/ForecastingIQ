import io
import asyncio
import os
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from typing import Optional

# ⚙️ Central config — see config.py. Everything environment-dependent
# (Firebase key path, CORS, dev password, Paymob keys, quick-train mode)
# is read from there so this file never hardcodes anything.
from config import settings, PLANS

# 📋 Logging + optional Sentry error tracking — set up before anything else
# so every module below can just do logging.getLogger("forecastiq").
from logging_setup import setup_logging
logger = setup_logging()

# 🔐 Firebase
import firebase_admin
from firebase_admin import credentials, auth, firestore

# 💳 Paymob (blank/inactive until keys are set in .env — see paymob.py)
import paymob
from paymob import PaymobNotConfigured

# 💰 Subscriptions & usage quotas
import billing
from billing import QuotaExceeded

# 🚦 Rate limiting (independent of subscription quotas)
from ratelimit import enforce_train_rate_limit, enforce_forecast_rate_limit

# 🏋️ Background training queue (so /train never blocks a request)
import jobs
from jobs import TrainingJobQueue, JobStatus


# =============================================================================
# 🔥  INIT FIREBASE
# =============================================================================
# FIREBASE_CRED_PATH (from .env, default "firebase_key.json") must point to
# your service-account JSON file. Get it from:
# Firebase Console → Project Settings → Service Accounts → Generate new private key.
# NEVER commit this file — it's already in .gitignore.
#
# Some hosts (Render) let you upload this as a real "secret file" directly.
# Others (Railway and most others) only offer plain environment variables —
# for those, base64-encode the JSON file's contents into one env var,
# FIREBASE_KEY_B64, and this block decodes it into a real file on startup
# before anything else runs. If FIREBASE_CRED_PATH already exists as an
# actual file (Render's case, or running locally), this is skipped entirely.
if not os.path.exists(settings.FIREBASE_CRED_PATH) and os.getenv("FIREBASE_KEY_B64"):
    with open(settings.FIREBASE_CRED_PATH, "wb") as f:
        f.write(base64.b64decode(os.environ["FIREBASE_KEY_B64"]))

if not os.path.exists(settings.FIREBASE_CRED_PATH):
    raise FileNotFoundError(
        f"Firebase service-account file not found at "
        f"'{settings.FIREBASE_CRED_PATH}'. Either place the real file there "
        f"(or point FIREBASE_CRED_PATH in your .env to its location), or — "
        f"on hosts without file upload (e.g. Railway) — set FIREBASE_KEY_B64 "
        f"to the base64-encoded contents of the file instead."
    )

cred = credentials.Certificate(settings.FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred)

db = firestore.client()
logger.info(f"Firebase initialized (environment={settings.ENVIRONMENT})")

security = HTTPBearer(auto_error=False)


def _save_trained_model(uid: str, bundle: dict):
    """Hook called by the training job queue once a job finishes successfully."""
    upload_model(uid, bundle)


training_queue = TrainingJobQueue(
    max_concurrent=settings.MAX_CONCURRENT_TRAINING_JOBS,
    db=db,
    on_complete=_save_trained_model,
)

# =============================================================================
# 📧  GMAIL CONFIG
# =============================================================================
GMAIL_SENDER   = settings.GMAIL_SENDER
GMAIL_APP_PASS = settings.GMAIL_APP_PASS

def send_forecast_email(to_email: str, forecast_csv: str, months: int, metrics: dict):
    if not settings.email_enabled:
        logger.info("Email delivery skipped — GMAIL_SENDER/GMAIL_APP_PASS not set in .env")
        return
    msg = MIMEMultipart()
    msg["From"]    = GMAIL_SENDER
    msg["To"]      = to_email
    msg["Subject"] = f"📊 Your {months}-Month Sales Forecast is Ready"

    body = f"""
<html><body style="font-family:Arial,sans-serif;background:#03040a;color:#f1f5f9;padding:32px;">
  <div style="max-width:560px;margin:0 auto;background:#0b0f1e;border:1px solid rgba(124,58,255,.25);border-radius:16px;padding:32px;">
    <h2 style="color:#7c3aff;margin-bottom:4px;">Sales Forecast Ready 🚀</h2>
    <p style="color:#64748b;font-size:13px;margin-bottom:24px;">Your {months}-month forecast has been generated successfully.</p>
    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:24px;">
      <tr style="border-bottom:1px solid rgba(255,255,255,.06);">
        <td style="padding:10px 0;color:#64748b;">Model</td>
        <td style="padding:10px 0;color:#f1f5f9;text-align:right;"><strong>{metrics.get('model_name','—')}</strong></td>
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,.06);">
        <td style="padding:10px 0;color:#64748b;">Val RMSE</td>
        <td style="padding:10px 0;color:#34d399;text-align:right;"><strong>{round(metrics.get('val_rmse',0),4)}</strong></td>
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,.06);">
        <td style="padding:10px 0;color:#64748b;">Baseline RMSE</td>
        <td style="padding:10px 0;color:#f1f5f9;text-align:right;">{round(metrics.get('baseline_rmse',0),4)}</td>
      </tr>
      <tr>
        <td style="padding:10px 0;color:#64748b;">Forecast Horizon</td>
        <td style="padding:10px 0;color:#06b6d4;text-align:right;"><strong>{months} months</strong></td>
      </tr>
    </table>
    <p style="color:#64748b;font-size:12px;">The full forecast is attached as a CSV file.</p>
    <p style="color:#64748b;font-size:11px;margin-top:24px;border-top:1px solid rgba(255,255,255,.06);padding-top:16px;">AI Demand Forecast API · Sent automatically after forecast generation</p>
  </div>
</body></html>
"""
    msg.attach(MIMEText(body, "html"))

    attachment = MIMEBase("application", "octet-stream")
    attachment.set_payload(forecast_csv.encode("utf-8"))
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", f"attachment; filename=forecast_{months}mo.csv")
    msg.attach(attachment)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_SENDER, GMAIL_APP_PASS)
        server.sendmail(GMAIL_SENDER, to_email, msg.as_string())


# =============================================================================
# ☁️  CLOUD MODELS DIRECTORY — مخزّنة في Firestore (base64) بدل Firebase Storage
# =============================================================================
# Firestore متاح مجاني على Spark plan (بدون billing account)، عكس Cloud Storage
# اللي بقى يطلب Blaze حتى لإنشاء الـ default bucket. بما إن موديلات Ridge/XGBoost
# بتاعتنا صغيرة، تخزينها كـ base64 string جوه document أبسط وأرخص.
#
# ⚠️ Firestore عنده حد أقصى 1 MiB لكل document. base64 بيكبّر حجم البيانات
# بنسبة ~33%، فاحنا حاطين سقف أمان عند 700KB للموديل الخام (raw bytes) قبل
# التحويل، عشان نضمن إننا تحت الـ 1 MiB بهامش كافي لباقي الـ fields.
MODEL_SIZE_LIMIT_BYTES = 700_000

def upload_model(uid: str, bundle: dict):
    """يحفظ الـ model كـ base64 string في Firestore تحت models/{uid}"""
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    raw_bytes = buf.getvalue()

    if len(raw_bytes) > MODEL_SIZE_LIMIT_BYTES:
        raise HTTPException(
            413,
            f"Trained model is {len(raw_bytes) / 1024:.0f}KB — too large to store in Firestore "
            f"(safe limit ~{MODEL_SIZE_LIMIT_BYTES / 1024:.0f}KB after base64 encoding). "
            "Reduce model complexity, or switch back to Firebase Storage (requires Blaze plan)."
        )

    encoded = base64.b64encode(raw_bytes).decode("ascii")
    db.collection("models").document(uid).set({
        "blob":       encoded,
        "size_bytes": len(raw_bytes),
        "updated_at": firestore.SERVER_TIMESTAMP,
    })


def download_model(uid: str):
    """يحمّل الـ model من Firestore ويرجّعه كـ dict bundle، أو None لو مفيش"""
    doc = db.collection("models").document(uid).get()
    if not doc.exists:
        return None

    data = doc.to_dict()
    raw_bytes = base64.b64decode(data["blob"])
    return joblib.load(io.BytesIO(raw_bytes))


def delete_model_cloud(uid: str) -> bool:
    """يحذف الـ model المحفوظ في Firestore"""
    doc_ref = db.collection("models").document(uid)
    if not doc_ref.get().exists:
        return False
    doc_ref.delete()
    return True


# =============================================================================
# 📊  FORECAST HISTORY — يحفظ نتيجة كل /forecast في Firestore
# =============================================================================
def save_forecast_to_firestore(
    uid: str,
    email: str,
    months: int,
    metrics: dict,
    preds_df: pd.DataFrame,
    historical_df: pd.DataFrame,
):
    """
    يحفظ نتيجة الـ forecast كـ document جديد تحت:
    users/{uid}/forecasts/{auto_id}
    بيحتوي الـ metrics + الـ predictions + نسخة خفيفة من الـ historical data
    (month, product_id, number_of_product_purchases) عشان insights.html يقدر
    يبني الصفحة من Firestore لوحده من غير ما يحتاج يرجع يرفع الملف الأصلي تاني.
    """
    hist_snapshot = historical_df[["month", "product_id", "number_of_product_purchases"]].copy()
    hist_snapshot["month"] = hist_snapshot["month"].dt.strftime("%Y-%m")

    doc_ref = db.collection("users").document(uid).collection("forecasts").document()
    doc_ref.set({
        "owner_email":   email,
        "created_at":    firestore.SERVER_TIMESTAMP,
        "months":        months,
        "model_name":    metrics.get("model_name"),
        "train_rmse":    metrics.get("train_rmse"),
        "val_rmse":      metrics.get("val_rmse"),
        "baseline_rmse": metrics.get("baseline_rmse"),
        "predictions":   preds_df.to_dict(orient="records"),
        "historical":    hist_snapshot.to_dict(orient="records"),
    })


# =============================================================================
# 🚀  FastAPI App
# =============================================================================
app = FastAPI(
    title="AI Demand Forecast API 🚀",
    description="Secure SaaS Forecasting API — train once, forecast anytime",
    version="5.0.0",
    swagger_ui_parameters={"persistAuthorization": True},
)

# NOTE: a wildcard origin ("*") combined with allow_credentials=True is
# actually rejected by browsers (the CORS spec forbids that combination
# when credentials are involved), so the old config silently broke
# cross-origin requests that relied on cookies/credentials. We now read
# the allow-list from ALLOWED_ORIGINS in .env — "*" for easy local dev,
# a comma-separated list of real domains in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 🌐  SERVE THE FRONTEND — same app, same origin, local or deployed
# =============================================================================
# Previously the HTML files weren't served by the backend at all, and one of
# them pointed at a hardcoded Railway URL while another pointed at
# 127.0.0.1 — so the two frontends only worked in different, contradictory
# setups. Now both files read their API base from `window.location.origin`
# (see the small inline patch at the top of their <script> — same file,
# no separate build step), and FastAPI serves everything from one origin,
# so "run locally" and "deploy to a server" both just work with zero edits.
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PAGES = {
    "/": "Sales-Forecast-app.html",
    "/app": "Sales-Forecast-app.html",
    "/dashboard": "dashboard.html",
    "/about": "about.html",
    "/guide": "user_guide.html",
}

for route_path, filename in FRONTEND_PAGES.items():
    full_path = os.path.join(FRONTEND_DIR, filename)

    def _make_handler(p=full_path):
        async def _handler():
            return FileResponse(p, media_type="text/html")
        return _handler

    app.get(route_path, include_in_schema=False)(_make_handler())


@app.get("/health", summary="Health check", include_in_schema=False)
def health_check():
    return JSONResponse({
        "status": "ok",
        "environment": settings.ENVIRONMENT,
        "email_enabled": settings.email_enabled,
        "paymob_configured": settings.paymob_configured,
        "dev_bypass_enabled": settings.dev_bypass_enabled,
    })


@app.on_event("startup")
async def _start_background_workers():
    training_queue.start_workers(num_workers=settings.MAX_CONCURRENT_TRAINING_JOBS)


@app.exception_handler(Exception)
async def _log_unhandled_exceptions(request: Request, exc: Exception):
    # Anything that reaches here is a bug, not an expected HTTPException —
    # log it with a full traceback (and forward to Sentry if configured, see
    # logging_setup.py) instead of it silently vanishing into a 500 response
    # that only the caller ever sees.
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. This has been logged."},
    )

# =============================================================================
# 🔐  VERIFY USER  (Firebase token — with an optional developer bypass)
# =============================================================================
DEV_UID = "dev-local-tester"

def verify_user(
    credential: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_dev_password: Optional[str] = Header(default=None, alias="X-Dev-Password"),
    dev_key: Optional[str] = Query(default=None),
):
    """
    Normal path: verifies the Firebase ID token exactly as before.

    Developer bypass: if DEV_ACCESS_PASSWORD is set in .env (and
    ENVIRONMENT is NOT "production"), you can skip Firebase auth entirely
    by sending that password as either:
      - header  X-Dev-Password: <password>
      - query   ?dev_key=<password>
    This is meant purely for trying the SaaS locally without wiring up a
    real Firebase login. All requests using it share one fixed pseudo-user
    ("dev-local-tester"), so trained models/forecasts persist between calls
    just like a real account would.
    """
    supplied_dev_password = x_dev_password or dev_key
    if settings.dev_bypass_enabled and supplied_dev_password:
        if supplied_dev_password == settings.DEV_ACCESS_PASSWORD:
            logger.warning(
                "🔑 DEV PASSWORD BYPASS USED — request authenticated as "
                f"'{DEV_UID}' without a real Firebase login. This is only "
                "safe in development; make sure DEV_ACCESS_PASSWORD is "
                "removed before going to production."
            )
            return {"uid": DEV_UID, "email": "dev@local.test", "is_dev_bypass": True}
        raise HTTPException(status_code=401, detail="Incorrect developer password")

    if credential is None:
        raise HTTPException(
            status_code=401,
            detail="Missing credentials — provide a Firebase Authorization "
                   "bearer token, or the developer password (see .env.example).",
        )

    try:
        decoded_token = auth.verify_id_token(credential.credentials, check_revoked=False)
        return decoded_token
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Token expired — please refresh the page or login again"
        )
    except auth.InvalidIdTokenError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {e}"
        )


# =============================================================================
# 📄  SWAGGER DOCS — auto-authorize from localStorage token
# =============================================================================
@app.get("/docs", include_in_schema=False)
async def custom_docs():
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="My API"
    ).body.decode("utf-8")

    wrap_script = """
<script src="https://www.gstatic.com/firebasejs/8.10.0/firebase-app.js"></script>
<script src="https://www.gstatic.com/firebasejs/8.10.0/firebase-auth.js"></script>
<script>
(function(){
    // ── Init Firebase ──────────────────────────────────────────────────────
    var firebaseConfig = {
        apiKey: "AIzaSyBTi0hycT_nCgThOoLDLDfXhCuLWeKcPMU",
        authDomain: "sales-forecasting-75f26.firebaseapp.com"
    };
    if (!firebase.apps.length) firebase.initializeApp(firebaseConfig);

    // currentUser: مرجع للـ Firebase user الحالي عشان نجيب منه token fresh في أي وقت
    var currentUser = null;

    // getToken: بيجيب token fresh من Firebase مباشرة (مش من localStorage)
    // forceRefresh=false: يستخدم الـ cache لو التوكن لسه صالح (< 1 hour)
    function getToken(forceRefresh) {
        if (!currentUser) return Promise.resolve(null);
        return currentUser.getIdToken(forceRefresh || false);
    }

    function applyTokenToSwagger(token) {
        if (!token || !window.ui) return;
        try {
            window.ui.authActions.authorize({
                HTTPBearer: {
                    name: "HTTPBearer", value: token,
                    schema: {type: "http", scheme: "bearer"}
                }
            });
        } catch(e) { console.warn("Swagger authorize error:", e); }
    }

    // ── Auth state listener ─────────────────────────────────────────────────
    firebase.auth().onAuthStateChanged(function(user) {
        if (!user) {
            currentUser = null;
            console.warn("⚠️ No Firebase user — please login first.");
            return;
        }
        currentUser = user;
        console.log("🔥 Firebase user:", user.email, "| UID:", user.uid);

        // جيب token fresh وحدّث الـ Swagger
        getToken(true).then(function(token) {
            applyTokenToSwagger(token);
            console.log("✅ Swagger authorized for:", user.email);
        });

        // جدد التوكن كل 50 دقيقة (قبل انتهاء الـ 60 دقيقة)
        setInterval(function() {
            getToken(true).then(function(token) {
                applyTokenToSwagger(token);
                console.log("🔄 Token proactively refreshed for:", user.email);
            });
        }, 50 * 60 * 1000);
    });

    // ── Swagger Bundle wrapper ──────────────────────────────────────────────
    var waitForBundle = setInterval(function(){
        if (typeof SwaggerUIBundle === "undefined") return;
        clearInterval(waitForBundle);

        var _Orig = SwaggerUIBundle;
        window.SwaggerUIBundle = function(cfg) {
            var _ri = cfg.requestInterceptor;

            // ⬇ كل request: اجيب token fresh من Firebase (مش من localStorage)
            // ده بيضمن إن كل request بيتبعت بتوكن صالح
            cfg.requestInterceptor = function(req) {
                // بنرجع promise — Swagger بيدعم async interceptors
                return getToken(false).then(function(token) {
                    if (token) {
                        req.headers["Authorization"] = "Bearer " + token;
                    }
                    return _ri ? _ri(req) : req;
                });
            };

            var _oc = cfg.onComplete;
            cfg.onComplete = function() {
                // لو التوكن جاهز authorize فورًا
                getToken(false).then(function(token) {
                    applyTokenToSwagger(token);
                    console.log("✅ Swagger auto-authorized on load");
                });
                if (_oc) _oc();
            };

            var instance = _Orig(cfg);
            window.ui = instance;
            return instance;
        };
        Object.keys(_Orig).forEach(function(k) {
            try { window.SwaggerUIBundle[k] = _Orig[k]; } catch(e) {}
        });
    }, 30);
})();
</script>"""

    last_script_pos = html.rfind("<script>")
    html = html[:last_script_pos] + wrap_script + "\n" + html[last_script_pos:]
    return HTMLResponse(html)


# =============================================================================
# CONSTANTS
# =============================================================================
REQUIRED_COLUMNS = {
    "month",
    "product_id",
    "number_of_product_purchases",
    "number_of_times_added_to_cart",
    "number_of_times_add_followed_by_purchase",
    "number_of_times_add_followed_by_no_purchase",
}


# =============================================================================
# HELPERS
# =============================================================================
async def read_upload_within_limit(file: UploadFile) -> bytes:
    """
    Reads an UploadFile's contents in chunks, aborting as soon as the total
    exceeds settings.MAX_UPLOAD_SIZE_MB — instead of trusting the
    Content-Length header (which a client can simply lie about) or reading
    the whole body into memory before checking its size (which lets a huge
    upload spike memory/CPU before validation ever runs).
    """
    limit = settings.max_upload_size_bytes
    chunk_size = 1024 * 1024  # 1 MB per chunk
    total = 0
    chunks = []
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                413,
                f"File too large — max {settings.MAX_UPLOAD_SIZE_MB}MB allowed."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def validate_and_load(contents: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV error: {exc}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(400, f"Missing columns: {sorted(missing)}")

    df = df.drop_duplicates()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    if df["month"].isna().any():
        raise HTTPException(400, "Invalid dates in 'month' column")

    return df.sort_values(["product_id", "month"]).reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["conversion_rate"] = (
        df["number_of_times_add_followed_by_purchase"]
        / df["number_of_times_added_to_cart"].replace(0, np.nan)
    )
    df["cart_drop_rate"] = (
        df["number_of_times_add_followed_by_no_purchase"]
        / df["number_of_times_added_to_cart"].replace(0, np.nan)
    )
    return df.drop(columns=[
        "number_of_times_add_followed_by_purchase",
        "number_of_times_add_followed_by_no_purchase",
    ])


FORECAST_HORIZON_OPTIONS = [3, 6, 9, 12]

def run_forecast(df: pd.DataFrame, bundle: dict, months: int = 3) -> pd.DataFrame:
    if months not in FORECAST_HORIZON_OPTIONS:
        raise ValueError(f"months must be one of {FORECAST_HORIZON_OPTIONS}")

    model     = bundle["model"]
    BEST_LAGS = bundle["lags"]
    BEST_ROLL = bundle["roll"]

    last_month      = df["month"].max()
    forecast_months = [
        last_month + pd.DateOffset(months=i)
        for i in range(1, months + 1)
    ]

    future_predictions = []

    for product in df["product_id"].unique():
        product_df = df[df["product_id"] == product]
        if len(product_df) < BEST_LAGS:
            continue

        hist       = product_df.iloc[-BEST_LAGS:]
        lag_values = hist["number_of_product_purchases"].values.tolist()
        conv_rate  = hist["conversion_rate"].iloc[-1]
        drop_rate  = hist["cart_drop_rate"].iloc[-1]

        for step in range(months):
            row = {f"lag_{i+1}": lag_values[i] for i in range(BEST_LAGS)}
            row["product_id"] = product

            valid_lags = [v for v in lag_values[:BEST_ROLL] if not np.isnan(v)]
            row[f"rolling_mean_{BEST_ROLL}"] = np.mean(valid_lags) if valid_lags else np.nan
            row[f"rolling_std_{BEST_ROLL}"]  = np.std(valid_lags) if len(valid_lags) > 1 else np.nan
            row["conversion_rate"] = conv_rate
            row["cart_drop_rate"]  = drop_rate

            pred = max(0.0, round(float(model.predict(pd.DataFrame([row]))[0]), 2))

            future_predictions.append({
                "product_id":          product,
                "forecast_month":      forecast_months[step].strftime("%Y-%m"),
                "predicted_purchases": pred,
            })

            lag_values = [pred] + lag_values[:-1]

    return pd.DataFrame(future_predictions)


# =============================================================================
# 🏋️  TRAIN ENDPOINT — enqueues a background training job (never blocks)
# =============================================================================
async def _quota_check(user: dict, kind: str, months: int = None):
    """
    Skips billing entirely for the developer bypass account — see billing.py.
    For kind="forecast", also enforces the requesting plan's forecast-horizon
    ceiling (e.g. the $30 plan can only forecast up to 3 months ahead) —
    checked BEFORE incrementing usage, so a rejected request never counts
    against the user's quota.
    """
    if user.get("is_dev_bypass"):
        return
    try:
        if kind == "forecast" and months is not None:
            sub = await asyncio.to_thread(billing.get_subscription, db, user["uid"])
            if not billing.is_subscription_active(sub):
                raise QuotaExceeded(
                    "No active subscription. Choose a plan at /billing/plans "
                    "and subscribe via /billing/subscribe to use this feature."
                )
            plan = PLANS[sub["plan"]]
            if months > plan["max_forecast_months"]:
                raise QuotaExceeded(
                    f"Your '{plan['name']}' plan allows forecasting up to "
                    f"{plan['max_forecast_months']} months ahead — you "
                    f"requested {months}. Upgrade your plan at "
                    f"/billing/plans to forecast further out."
                )

        # billing.check_and_increment_quota does a synchronous Firestore
        # transaction — run it in a thread so it can't stall the asyncio
        # event loop (and every other in-flight request) while it waits
        # on the network.
        await asyncio.to_thread(billing.check_and_increment_quota, db, user["uid"], kind)
    except QuotaExceeded as exc:
        raise HTTPException(402, str(exc))


@app.post(
    "/train",
    summary="Upload CSV → queue a training job → poll for the result",
    response_description="A job_id to poll via GET /train/status/{job_id}",
)
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file with historical sales data"),
    user=Depends(verify_user),
):
    """
    **Flow (now asynchronous):**
    1. Validate the CSV and feature-engineer it (fast, done inline)
    2. Enqueue a training job — a bounded worker pool (see jobs.py) runs the
       actual grid search in the background, so this request returns
       immediately instead of holding the connection open for minutes
    3. Poll `GET /train/status/{job_id}` until status is "done" or "failed"
    4. Once done, the model is already saved — call `/forecast` normally

    This also means one user (or a bot) can no longer take the server down
    by firing off many concurrent /train calls: only
    `MAX_CONCURRENT_TRAINING_JOBS` (see .env) actually run grid searches at
    once, everything else waits in the queue.
    """
    # ── Rate limit: independent of subscription quota, stops raw hammering ──
    enforce_train_rate_limit(user["uid"], per_hour=settings.RATE_LIMIT_TRAIN_PER_HOUR)

    # ── Subscription quota: has this user paid for another training run? ───
    await _quota_check(user, "training")

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    contents = await read_upload_within_limit(file)

    # ── Validate + feature engineer inline (fast — no need to queue this part) ──
    df = validate_and_load(contents)
    df = engineer_features(df)

    job_id = training_queue.submit(user["uid"], user.get("email", "unknown"), df)
    logger.info(f"Training job {job_id} queued for uid={user['uid']}")

    return JSONResponse({
        "message": "Training job queued. Poll GET /train/status/{job_id} for progress.",
        "job_id": job_id,
        "status": JobStatus.QUEUED,
    })


@app.get("/train/status/{job_id}", summary="Check a training job's progress/result")
def train_status(job_id: str, user=Depends(verify_user)):
    if not training_queue.owns(job_id, user["uid"]):
        raise HTTPException(404, "No such training job for your account.")
    status = training_queue.get_status(job_id)
    if not status:
        raise HTTPException(404, "No such training job for your account.")
    return JSONResponse(status)


# =============================================================================
# 🚀  FORECAST ENDPOINT — يلود الـ model المحفوظ ويعمل predict مباشرة
# =============================================================================
@app.post(
    "/forecast",
    summary="Upload CSV → load your saved model → get forecast",
    response_description="CSV file with predicted purchases per product",
)
async def forecast_endpoint(
    file: UploadFile = File(..., description="CSV file with historical sales data"),
    months: int = Query(3, description="Forecast horizon in months", enum=FORECAST_HORIZON_OPTIONS),
    user=Depends(verify_user),
):
    """
    **Flow:**
    1. التحقق إن اليوزر عمل /train قبل كده
    2. تحميل الـ model المحفوظ من disk
    3. استقبال الـ CSV الجديد وعمل Feature engineering عليه
    4. **Forecast** مباشرة بدون أي retrain
    5. يرجّع CSV بالـ predictions

    > ⚠️ لازم تعمل `/train` الأول قبل ما تستخدم هذا الـ endpoint
    """
    # ── Rate limit: independent of subscription quota ───────────────────────
    enforce_forecast_rate_limit(user["uid"], per_minute=settings.RATE_LIMIT_FORECAST_PER_MINUTE)

    # ── Subscription quota + forecast-horizon check ──────────────────────────
    await _quota_check(user, "forecast", months=months)

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    # ── 1. Read the upload (size-capped) before touching Firestore at all ──
    # so an oversized/malicious upload gets rejected without wasting a
    # network round-trip on a model that might not even end up being used.
    contents = await read_upload_within_limit(file)

    # ── 2. Load user's saved model ────────────────────────────────────────
    bundle = await asyncio.to_thread(download_model, user["uid"])
    if not bundle:
        raise HTTPException(
            404,
            "No trained model found for your account. "
            "Please call POST /train first with your historical data."
        )

    # ── 3. Load & validate ────────────────────────────────────────────────
    df = validate_and_load(contents)

    # ── 4. Feature engineering ────────────────────────────────────────────
    df = engineer_features(df)

    # ── 5. Forecast ───────────────────────────────────────────────────────
    preds_df = run_forecast(df, bundle, months=months)

    if preds_df.empty:
        raise HTTPException(422, "Not enough data per product to forecast")

    # ── 6. Return CSV ─────────────────────────────────────────────────────
    stream = io.StringIO()
    preds_df.to_csv(stream, index=False)
    stream.seek(0)

    metrics = bundle["metrics"]

    # ── 6.5 Save forecast to Firestore (non-blocking — never breaks the CSV download) ──
    try:
        await asyncio.to_thread(
            save_forecast_to_firestore,
            user["uid"],
            user.get("email", "unknown"),
            months,
            metrics,
            preds_df,
            df,
        )
    except Exception as fs_exc:
        logger.warning(f"Firestore save failed (non-fatal): {fs_exc}", exc_info=True)

    # ── 6. Send email (non-blocking) ─────────────────────────────────────────
    user_email = user.get("email")
    if user_email:
        try:
            await asyncio.to_thread(
                send_forecast_email,
                user_email,
                stream.getvalue(),
                months,
                metrics,
            )
        except Exception as mail_exc:
            logger.warning(f"Email send failed (non-fatal): {mail_exc}", exc_info=True)

    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=forecast_{user['uid']}.csv",
            "X-Model-Name":    metrics["model_name"],
            "X-Train-RMSE":    str(round(metrics["train_rmse"],    4)),
            "X-Val-RMSE":      str(round(metrics["val_rmse"],      4)),
            "X-Baseline-RMSE": str(round(metrics["baseline_rmse"], 4)),
            "X-Best-Lags":     str(bundle["lags"]),
            "X-Best-Roll":     str(bundle["roll"]),
            "X-Forecast-Months": str(months),
        },
    )


# =============================================================================
# 🗑️  DELETE MODEL ENDPOINT — يحذف الـ model المحفوظ للـ user
# =============================================================================
@app.delete(
    "/model",
    summary="Delete your saved model",
)
async def delete_model(user=Depends(verify_user)):
    """
    يحذف الـ model المحفوظ للـ user — بعد كده لازم يعمل /train تاني.
    مفيد لو اليوزر عايز يعيد الـ training على داتا جديدة من الأساس.
    """
    deleted = await asyncio.to_thread(delete_model_cloud, user["uid"])
    if not deleted:
        raise HTTPException(404, "No model found to delete.")
    return JSONResponse({"message": "✅ Model deleted. You can now retrain with new data."})


# =============================================================================
# 📊  METRICS ENDPOINT (PROTECTED)
# =============================================================================
@app.get("/metrics", summary="Get your saved model's metrics")
def get_metrics(user=Depends(verify_user)):
    """
    يرجّع الـ metrics الخاصة بالـ model المحفوظ للـ user.
    """
    bundle = download_model(user["uid"])
    if not bundle:
        raise HTTPException(
            404,
            "No trained model found. Please call POST /train first."
        )

    metrics = bundle["metrics"]

    return JSONResponse({
        "model_name":    metrics["model_name"],
        "owner_email":   bundle.get("owner_email", "unknown"),
        "train_rmse":    round(metrics["train_rmse"],    4),
        "val_rmse":      round(metrics["val_rmse"],      4),
        "baseline_rmse": round(metrics["baseline_rmse"], 4),
        "best_lags":     bundle["lags"],
        "best_roll":     bundle["roll"],
    })


# =============================================================================
# 📜  FORECAST HISTORY ENDPOINTS — لصفحة insights.html
# =============================================================================
@app.get("/forecasts", summary="List your saved forecast runs (for the Insights history dropdown)")
def list_forecasts(
    user=Depends(verify_user),
    limit: int = Query(20, ge=1, le=100, description="Max number of past forecasts to return"),
):
    """
    يرجّع قائمة مختصرة لآخر forecasts المحفوظة لليوزر (الأحدث أولاً) —
    من غير الـ predictions/historical الكاملة، عشان الـ dropdown يكون سريع.
    """
    docs = (
        db.collection("users").document(user["uid"])
          .collection("forecasts")
          .order_by("created_at", direction=firestore.Query.DESCENDING)
          .limit(limit)
          .stream()
    )
    items = []
    for d in docs:
        data = d.to_dict()
        created_at = data.get("created_at")
        items.append({
            "id":         d.id,
            "created_at": created_at.isoformat() if created_at else None,
            "months":     data.get("months"),
            "model_name": data.get("model_name"),
            "val_rmse":   data.get("val_rmse"),
        })
    return JSONResponse({"forecasts": items})


@app.get("/forecasts/{forecast_id}", summary="Get full data for one saved forecast (predictions + historical)")
def get_forecast(forecast_id: str, user=Depends(verify_user)):
    """
    يرجّع forecast واحد بالتفصيل (predictions + historical snapshot) —
    ده اللي insights.html بيستخدمه لبناء الصفحة من غير الحاجة لـ sessionStorage.
    """
    doc = (
        db.collection("users").document(user["uid"])
          .collection("forecasts").document(forecast_id).get()
    )
    if not doc.exists:
        raise HTTPException(404, "Forecast not found")

    data = doc.to_dict()
    created_at = data.get("created_at")
    return JSONResponse({
        "id":           doc.id,
        "created_at":   created_at.isoformat() if created_at else None,
        "months":       data.get("months"),
        "model_name":   data.get("model_name"),
        "val_rmse":     data.get("val_rmse"),
        "predictions":  data.get("predictions", []),
        "historical":   data.get("historical", []),
    })


# =============================================================================
# 💰  SUBSCRIPTION / BILLING ENDPOINTS
# =============================================================================
# Three fixed plans (see config.PLANS: forecast_3mo / forecast_6mo /
# forecast_12mo — priced by forecast horizon, not access duration). Paying via
# Paymob activates the subscription automatically through the webhook —
# INACTIVE until you add real Paymob credentials to .env (see paymob.py).
@app.get("/billing/plans", summary="List available subscription plans")
def list_plans():
    return JSONResponse({
        "plans": [
            {"id": plan_id, **plan}
            for plan_id, plan in PLANS.items()
            # test_plan is a $0.50 plan for exercising the Paymob checkout
            # flow without paying full price — hide it from real customers
            # by only ever showing it outside production. Remove this
            # whole entry from PLANS in config.py once testing is done.
            if plan_id != "test_plan" or not settings.is_production
        ],
        "paymob_configured": settings.paymob_configured,
    })


@app.get("/billing/status", summary="Get your current subscription and usage")
def billing_status(user=Depends(verify_user)):
    if user.get("is_dev_bypass"):
        return JSONResponse({
            "plan": "dev-bypass",
            "status": "active",
            "note": "Developer bypass — subscription/quota checks are skipped entirely for this account.",
        })

    sub = billing.get_subscription(db, user["uid"])
    active = billing.is_subscription_active(sub)
    response = {
        "plan": sub["plan"],
        "status": sub["status"] if active else "inactive",
        "expires_at": sub["expires_at"].isoformat() if sub.get("expires_at") else None,
    }
    if active:
        usage = billing.get_usage(db, user["uid"], sub["cycle_id"])
        plan = PLANS[sub["plan"]]
        response["usage"] = {
            "trainings_used": usage.get("trainings_used", 0),
            "trainings_limit": plan["max_trainings"],
            "forecasts_used": usage.get("forecasts_used", 0),
            "forecasts_limit": plan["max_forecasts"],
        }
    return JSONResponse(response)


class SubscribeRequest(BaseModel):
    plan_id: str  # "forecast_3mo" | "forecast_6mo" | "forecast_12mo"


@app.post("/billing/subscribe", summary="Start a Paymob checkout for a subscription plan")
async def subscribe(body: SubscribeRequest, user=Depends(verify_user)):
    if body.plan_id not in PLANS:
        raise HTTPException(400, f"Unknown plan_id. Choose one of: {', '.join(PLANS)}")

    if not settings.paymob_configured:
        raise HTTPException(
            503,
            "Payments aren't set up yet — add PAYMOB_API_KEY, "
            "PAYMOB_INTEGRATION_ID and PAYMOB_IFRAME_ID to your .env file."
        )

    plan = PLANS[body.plan_id]
    amount_egp = plan["price_usd"] * settings.USD_TO_EGP_RATE

    try:
        result = await paymob.create_payment_intent(
            amount_cents=int(round(amount_egp * 100)),
            billing_email=user.get("email", "unknown@local.test"),
        )
    except PaymobNotConfigured as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error(f"Paymob request failed for uid={user['uid']}: {exc}")
        raise HTTPException(502, f"Paymob request failed: {exc}")

    # Remember which plan this Paymob order is for, so the webhook (which
    # only gets Paymob's order id back, not our plan_id) knows what to
    # activate once payment succeeds.
    await asyncio.to_thread(
        lambda: db.collection("pending_orders").document(str(result["order_id"])).set({
            "uid": user["uid"],
            "plan_id": body.plan_id,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
    )

    return JSONResponse({
        "plan_id": body.plan_id,
        "amount_usd": plan["price_usd"],
        "amount_egp": round(amount_egp, 2),
        "order_id": result["order_id"],
        "iframe_url": result["iframe_url"],
    })


@app.post("/payment/webhook", summary="Paymob transaction webhook (requires PAYMOB_HMAC_SECRET)", include_in_schema=False)
async def paymob_webhook(payload: dict):
    if not settings.paymob_configured or not settings.PAYMOB_HMAC_SECRET:
        raise HTTPException(503, "Paymob webhook is not configured yet.")

    received_hmac = payload.get("hmac", "")
    transaction = payload.get("obj", {})

    try:
        valid = paymob.verify_hmac(received_hmac, transaction)
    except PaymobNotConfigured as exc:
        raise HTTPException(503, str(exc))

    if not valid:
        logger.warning("Paymob webhook received an invalid HMAC signature — request rejected.")
        raise HTTPException(400, "Invalid HMAC signature — request rejected.")

    order_id = transaction.get("order")

    if transaction.get("success"):
        order_doc = await asyncio.to_thread(
            lambda: db.collection("pending_orders").document(str(order_id)).get()
        )
        if not order_doc.exists:
            logger.error(f"Paymob webhook: no pending_orders record for order {order_id} — cannot activate a subscription.")
            return JSONResponse({"received": True, "warning": "No matching pending order found."})

        order_data = order_doc.to_dict()
        result = await asyncio.to_thread(
            billing.activate_subscription, db, order_data["uid"], order_data["plan_id"], str(order_id)
        )
        await asyncio.to_thread(
            lambda: db.collection("pending_orders").document(str(order_id)).delete()
        )
        logger.info(f"✅ Subscription activated for uid={order_data['uid']}: plan={order_data['plan_id']}, expires={result['expires_at']}")
    else:
        logger.info(f"Paymob payment failed/declined for order {order_id}: transaction id {transaction.get('id')}")

    return JSONResponse({"received": True})

# =============================================================================
# 🚀  LOCAL RUN — `python main.py`
# =============================================================================
# This block only matters when running locally. When deploying to a real
# server (Railway, Render, a VPS with gunicorn/uvicorn, Docker, etc.) you'll
# instead run something like:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
# and this block is simply never executed (the `if __name__` guard), so no
# code changes are needed to go from local -> server.
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting locally at http://{settings.HOST}:{settings.PORT}  (environment={settings.ENVIRONMENT})")
    print(f"   App:        http://{settings.HOST}:{settings.PORT}/")
    print(f"   Dashboard:  http://{settings.HOST}:{settings.PORT}/dashboard")
    print(f"   API docs:   http://{settings.HOST}:{settings.PORT}/docs")
    print(f"   Billing:    http://{settings.HOST}:{settings.PORT}/billing/plans  (paymob_configured={settings.paymob_configured})")
    if settings.dev_bypass_enabled:
        print(f"   🔑 Dev bypass ACTIVE — send header 'X-Dev-Password' to skip Firebase login.")
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=not settings.is_production)