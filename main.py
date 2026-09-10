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
from config import settings, PLANS, PLAN_FEATURE_EXPLANATIONS, ALL_FORECAST_HORIZON_OPTIONS, forecast_horizon_options

# 📋 Logging + optional Sentry error tracking — set up before anything else
# so every module below can just do logging.getLogger("forecastiq").
from logging_setup import setup_logging
logger = setup_logging()

# 🔐 Firebase
import firebase_admin
from firebase_admin import credentials, auth, firestore

# 💳 Paymob (blank/inactive until keys are set in .env — see paymob.py).
# Single merchant integration, single currency (settings.BASE_CURRENCY) —
# see config.DISPLAY_CURRENCIES for the customer-facing display-only
# currency list (does not change what's actually charged).
import paymob
from paymob import PaymobNotConfigured

# 💰 Subscriptions & usage quotas
import billing
from billing import QuotaExceeded, TrialAlreadyUsed

# 🚦 Rate limiting (independent of subscription quotas)
from ratelimit import enforce_train_rate_limit, enforce_forecast_rate_limit

# 🏋️ Background training queue (so /train never blocks a request)
import jobs
import forecast_store
from jobs import TrainingJobQueue, JobStatus

# 🧬 Automatic column detection so /train and /forecast work with any CSV
import schema


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
    save_model(uid, bundle, name=bundle.pop("requested_model_name", None))


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
#
# NOTE: each user can now save SEVERAL trained models (capped by their
# plan's max_models — see config.PLANS) instead of just one. Models live
# at users/{uid}/models/{model_id} instead of the old models/{uid} single
# document, so /train no longer overwrites a previous model — it adds a
# new one, and /forecast is told which one to use via ?model_id=.
MODEL_SIZE_LIMIT_BYTES = 700_000


def _models_collection(uid: str):
    return db.collection("users").document(uid).collection("models")


def count_models(uid: str) -> int:
    """Cheap-ish count of how many trained models this user currently has
    saved. Avoids Query.select([]) (unsupported/finicky on some Firestore
    client versions) — fetching just the ids is small and reliable."""
    return sum(1 for _ in _models_collection(uid).list_documents())


def save_model(uid: str, bundle: dict, name: str = None) -> str:
    """Saves a NEW trained model as base64 in Firestore under
    users/{uid}/models/{model_id} and returns the new model_id. Never
    overwrites an existing model."""
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
    metrics = bundle.get("metrics", {})
    doc_ref = _models_collection(uid).document()
    doc_ref.set({
        "name":          (name or "").strip() or metrics.get("model_name") or "Untitled model",
        "blob":          encoded,
        "size_bytes":    len(raw_bytes),
        "created_at":    firestore.SERVER_TIMESTAMP,
        "updated_at":    firestore.SERVER_TIMESTAMP,
        "owner_email":   bundle.get("owner_email", "unknown"),
        "model_name":    metrics.get("model_name"),
        "train_rmse":    metrics.get("train_rmse"),
        "val_rmse":      metrics.get("val_rmse"),
        "baseline_rmse": metrics.get("baseline_rmse"),
        "lags":          bundle.get("lags"),
        "roll":          bundle.get("roll"),
        "schema":        bundle.get("schema"),
    })
    return doc_ref.id


def list_models(uid: str) -> list[dict]:
    """Returns lightweight metadata (no blob) for every model this user has
    saved, newest first — used by the dashboard's model picker/list."""
    docs = _models_collection(uid).order_by(
        "created_at", direction=firestore.Query.DESCENDING
    ).stream()
    items = []
    for d in docs:
        data = d.to_dict()
        created_at = data.get("created_at")
        items.append({
            "id":            d.id,
            "name":          data.get("name") or data.get("model_name") or "Untitled model",
            "created_at":    created_at.isoformat() if created_at else None,
            "model_name":    data.get("model_name"),
            "train_rmse":    data.get("train_rmse"),
            "val_rmse":      data.get("val_rmse"),
            "baseline_rmse": data.get("baseline_rmse"),
            "schema":        data.get("schema"),
            "size_bytes":    data.get("size_bytes"),
        })
    return items


def download_model(uid: str, model_id: str):
    """يحمّل موديل معيّن من Firestore ويرجّعه كـ dict bundle، أو None لو مفيش"""
    doc = _models_collection(uid).document(model_id).get()
    if not doc.exists:
        return None

    data = doc.to_dict()
    raw_bytes = base64.b64decode(data["blob"])
    return joblib.load(io.BytesIO(raw_bytes))


def delete_model_cloud(uid: str, model_id: str) -> bool:
    """يحذف موديل معيّن محفوظ في Firestore"""
    doc_ref = _models_collection(uid).document(model_id)
    if not doc_ref.get().exists:
        return False
    doc_ref.delete()
    return True


# =============================================================================
# 📊  FORECAST HISTORY — يحفظ نتيجة كل /forecast في Postgres (Railway)
# =============================================================================
# Moved off Firestore: a saved forecast (predictions + historical snapshot)
# routinely exceeds Firestore's 1 MiB/document cap once there are many
# groups and/or a long horizon — that's exactly what silently broke "View
# Insights". See forecast_store.py for the Postgres table + queries.
def save_forecast_to_history(
    uid: str,
    email: str,
    months: int,
    metrics: dict,
    preds_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    sch: dict,
    model_id: str = None,
) -> str:
    """
    يحفظ نتيجة الـ forecast كـ صف جديد في جدول forecasts على Postgres —
    بيحتوي الـ metrics + الـ predictions + نسخة خفيفة من الـ historical data
    (date/group/target columns من الـ schema المحفوظة) عشان insights.html
    يقدر يبني الصفحة من غير ما يحتاج يرجع يرفع الملف الأصلي تاني.
    Column-agnostic — يشتغل مع أي schema تم اكتشافه وقت الـ train.
    Returns the new row's id so the caller can hand it straight to
    the Insights page (?forecast_id=...).
    """
    snapshot_cols = [sch["date_col"], sch["target_col"]] + ([sch["group_col"]] if sch["group_col"] else [])
    hist_snapshot = historical_df[snapshot_cols].copy()
    hist_snapshot[sch["date_col"]] = hist_snapshot[sch["date_col"]].dt.strftime("%Y-%m")

    return forecast_store.save_forecast(
        uid=uid,
        email=email,
        months=months,
        metrics=metrics,
        predictions=preds_df.to_dict(orient="records"),
        historical=hist_snapshot.to_dict(orient="records"),
        sch=sch,
        model_id=model_id,
    )


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
    # Browsers hide all response headers from JS by default except a
    # short safelist — X-Forecast-Id etc. need to be explicitly exposed
    # here or dashboard.html's fetch() can't read them at all, even
    # though they're clearly visible in the Network tab.
    expose_headers=["X-Forecast-Id", "X-Model-Name", "X-Train-RMSE", "X-Val-RMSE", "X-Baseline-RMSE", "X-Best-Lags", "X-Best-Roll", "X-Forecast-Months", "X-Forecast-Save-Error"],
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
    # Real, linkable/bookmarkable route for the pricing flow's "Plan Details"
    # step — reads ?plan=starter&mode=subscribe|trial from the URL itself
    # (see plan-details.html), so Pricing → Select Plan → Plan Details →
    # Subscribe → Payment is now actual navigation, not a modal overlay.
    "/plan-details": "plan-details.html",
    # Power-BI-style analytics dashboard for a saved forecast run — see
    # /forecasts and /forecasts/{id} (the data it's built from) and
    # dashboard.html's viewInsights() (the button that links here).
    "/insights": "insights.html",
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


@app.on_event("startup")
async def _init_postgres():
    # Creates the forecasts table if it doesn't exist yet — safe to run
    # on every boot. If DATABASE_URL isn't set, this raises loudly at
    # startup instead of failing silently the first time someone hits
    # "View Insights".
    await asyncio.to_thread(forecast_store.init_db)


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


def _optional_user(
    credential: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_dev_password: Optional[str] = Header(default=None, alias="X-Dev-Password"),
    dev_key: Optional[str] = Query(default=None),
) -> Optional[dict]:
    """Same as verify_user, but returns None instead of raising 401 when no
    credentials are supplied — for endpoints (like /billing/plans) that stay
    public but personalize their response when the caller happens to be
    logged in."""
    if credential is None and not (x_dev_password or dev_key):
        return None
    try:
        return verify_user(credential, x_dev_password, dev_key)
    except HTTPException:
        return None


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


def load_csv(contents: bytes) -> pd.DataFrame:
    """Parses the raw upload into a DataFrame — no assumption about its
    columns. Column detection/validation happens afterwards, separately,
    for /train (auto-detect + save a new schema) vs /forecast (validate
    against the schema saved at training time)."""
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV error: {exc}")

    if df.empty or len(df.columns) == 0:
        raise HTTPException(400, "The CSV appears to be empty.")

    return df.drop_duplicates()


def prepare_for_schema(df: pd.DataFrame, sch: dict) -> pd.DataFrame:
    """Coerces the date column to datetime and the numeric columns
    (target + numeric_features) to true numeric dtype, then sorts by
    (group_col, date_col) — shared by both /train (right after detecting a
    new schema) and /forecast (against the schema saved with the model).

    The numeric coercion matters even though schema.detect_schema() only
    classifies a column as numeric when ~90%+ of its values already parse
    as numbers — the remaining <10% (blanks, stray text, "N/A") can leave
    the column as `object` dtype, which sklearn's imputer/scaler can't
    handle (`ufunc 'isnan' not supported for object arrays`) even though
    pandas is happy to store it. Coercing here with errors="coerce" turns
    any leftover non-numeric values into NaN, which the training
    pipeline's IterativeImputer is built to handle already.
    """
    date_col, group_col = sch["date_col"], sch["group_col"]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise HTTPException(400, f"Invalid/unparseable dates in column '{date_col}'")

    for c in [sch["target_col"]] + sch["numeric_features"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[sch["target_col"]].isna().all():
        raise HTTPException(400, f"Target column '{sch['target_col']}' has no valid numeric values.")

    sort_cols = [group_col, date_col] if group_col else [date_col]
    return df.sort_values(sort_cols).reset_index(drop=True)


def validate_upload_against_schema(df: pd.DataFrame, sch: dict) -> None:
    """Raises a clear 400 if a /forecast upload doesn't have the columns
    the saved model was trained on (it doesn't need to detect anything —
    just confirm the expected columns are present)."""
    required = [sch["date_col"], sch["target_col"]] + ([sch["group_col"]] if sch["group_col"] else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            400,
            f"This CSV is missing column(s) the trained model expects: {missing}. "
            "Upload a file with the same columns you used for /train."
        )


FORECAST_HORIZON_OPTIONS = ALL_FORECAST_HORIZON_OPTIONS  # [3, 6, 12, 18, 24] — see config.py

def run_forecast(df: pd.DataFrame, bundle: dict, months: int = 3) -> pd.DataFrame:
    if months not in FORECAST_HORIZON_OPTIONS:
        raise ValueError(f"months must be one of {FORECAST_HORIZON_OPTIONS}")

    model  = bundle["model"]
    sch    = bundle["schema"]
    BEST_LAGS = bundle["lags"]
    BEST_ROLL = bundle["roll"]

    date_col   = sch["date_col"]
    group_col  = sch["group_col"]
    target_col = sch["target_col"]
    num_extra  = sch["numeric_features"]
    cat_extra  = sch["categorical_features"]

    group_key = group_col or "__single_series__"
    df = df.copy()
    if group_col is None:
        df[group_key] = "all"

    last_period      = df[date_col].max()
    forecast_periods = [last_period + pd.DateOffset(months=i) for i in range(1, months + 1)]

    future_predictions = []

    for group_value in df[group_key].unique():
        gdf = df[df[group_key] == group_value].sort_values(date_col)
        if len(gdf) < BEST_LAGS:
            continue

        hist       = gdf.iloc[-BEST_LAGS:]
        lag_values = hist[target_col].values.tolist()
        extra_num_vals = {c: float(hist[c].iloc[-1]) for c in num_extra}
        # Cast the same way training does (Smart_Za3bola.py fills NaN with
        # "missing" and casts to str before fitting the OneHotEncoder) —
        # predicting with a raw NaN or a non-string type here would silently
        # mismatch what the fitted encoder expects.
        extra_cat_vals = {
            c: (str(hist[c].iloc[-1]) if c in hist.columns and pd.notna(hist[c].iloc[-1]) else "missing")
            for c in cat_extra if c != group_col
        }

        for step in range(months):
            row = {f"lag_{i+1}": lag_values[i] for i in range(BEST_LAGS)}
            if group_col:
                row[group_col] = str(group_value)  # training casts group_col to str too — must match exactly

            valid_lags = [v for v in lag_values[:BEST_ROLL] if not np.isnan(v)]
            row[f"rolling_mean_{BEST_ROLL}"] = np.mean(valid_lags) if valid_lags else np.nan
            row[f"rolling_std_{BEST_ROLL}"]  = np.std(valid_lags) if len(valid_lags) > 1 else np.nan
            row.update(extra_num_vals)
            row.update(extra_cat_vals)

            pred = max(0.0, round(float(model.predict(pd.DataFrame([row]))[0]), 2))

            result_row = {
                "forecast_period": forecast_periods[step].strftime("%Y-%m"),
                "predicted_value": pred,
            }
            if group_col:
                result_row[group_col] = group_value
            future_predictions.append(result_row)

            lag_values = [pred] + lag_values[:-1]

    return pd.DataFrame(future_predictions)


# =============================================================================
# 🏋️  TRAIN ENDPOINT — enqueues a background training job (never blocks)
# =============================================================================
async def _quota_check(user: dict, kind: str, months: int = None, amount: int = 1, increment: bool = True):
    """
    Skips billing entirely for the developer bypass account — see billing.py.
    For kind="forecast_points" with `months` given, also enforces the
    requesting plan's forecast-horizon ceiling (e.g. Starter can only
    forecast up to 3 months ahead) — checked BEFORE incrementing usage, so
    a rejected request never counts against the user's quota.
    Pass increment=False to only run the horizon check without touching
    the usage counter (used for the early forecast-endpoint check, before
    we know how many forecasted data points the request will produce).
    """
    if user.get("is_dev_bypass"):
        return
    try:
        if kind == "forecast_points" and months is not None:
            sub = await asyncio.to_thread(billing.get_subscription, db, user["uid"])
            if not billing.is_subscription_active(sub):
                raise QuotaExceeded(
                    "No active subscription. Choose a plan at /billing/plans "
                    "and subscribe via /billing/subscribe to use this feature."
                )
            plan = PLANS[sub["plan"]]
            allowed = forecast_horizon_options(plan)
            if months not in allowed:
                raise QuotaExceeded(
                    f"Your '{plan['name']}' plan supports forecasting "
                    f"{', '.join(str(m) for m in allowed)} months ahead — you "
                    f"requested {months}. Upgrade your plan at "
                    f"/billing/plans for a longer forecast horizon."
                )

        if not increment:
            return

        # billing.check_and_increment_quota does a synchronous Firestore
        # transaction — run it in a thread so it can't stall the asyncio
        # event loop (and every other in-flight request) while it waits
        # on the network.
        await asyncio.to_thread(billing.check_and_increment_quota, db, user["uid"], kind, amount)
    except QuotaExceeded as exc:
        raise HTTPException(402, str(exc))


@app.post(
    "/train",
    summary="Upload CSV → queue a training job → poll for the result",
    response_description="A job_id to poll via GET /train/status/{job_id}",
)
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file with historical data — any columns"),
    date_column: Optional[str] = Query(None, description="Force which column is the date/time axis (auto-detected if omitted)"),
    group_column: Optional[str] = Query(None, description="Force which column splits the data into separate series, e.g. product_id (auto-detected if omitted; omit entirely for a single series)"),
    target_column: Optional[str] = Query(None, description="Force which numeric column to forecast (auto-detected if omitted)"),
    model_name: Optional[str] = Query(None, description="Optional display name to save this model under, e.g. 'Store 12 — monthly sales' (shown in the dashboard's model list)"),
    user=Depends(verify_user),
):
    """
    **Flow (now asynchronous):**
    1. Validate the CSV — works with ANY columns. We auto-detect which
       column is the date, which (if any) splits the data into separate
       series (e.g. product_id), and which numeric column to forecast —
       override any of those with date_column/group_column/target_column
       if you want to pin them explicitly instead of relying on
       auto-detection.
    2. Enqueue a training job — a bounded worker pool (see jobs.py) runs the
       actual grid search in the background, so this request returns
       immediately instead of holding the connection open for minutes
    3. Poll `GET /train/status/{job_id}` until status is "done" or "failed"
    4. Once done, the model is already saved — call `/forecast` normally
       with a CSV that has the SAME columns (the detected schema is saved
       with the model and reused, not re-detected, on /forecast)

    This also means one user (or a bot) can no longer take the server down
    by firing off many concurrent /train calls: only
    `MAX_CONCURRENT_TRAINING_JOBS` (see .env) actually run grid searches at
    once, everything else waits in the queue.
    """
    # ── Rate limit: independent of subscription quota, stops raw hammering ──
    enforce_train_rate_limit(user["uid"], per_hour=settings.RATE_LIMIT_TRAIN_PER_HOUR)

    # ── Subscription quota: has this user paid for another training run? ───
    await _quota_check(user, "training")

    # ── Saved-model cap: how many trained models does this plan allow you
    #    to keep at once? Checked BEFORE queueing the job so we never burn
    #    a training run (and its quota) on a model that can't be saved. ───
    if not user.get("is_dev_bypass"):
        sub = await asyncio.to_thread(billing.get_subscription, db, user["uid"])
        if billing.is_subscription_active(sub):
            plan = PLANS[sub["plan"]]
            existing = await asyncio.to_thread(count_models, user["uid"])
            if existing >= plan["max_models"]:
                raise HTTPException(
                    402,
                    f"Your '{plan['name']}' plan can keep at most {plan['max_models']} "
                    f"saved trained model(s) — you already have {existing}. Delete an "
                    f"existing model from the dashboard, or upgrade your plan at /billing/plans."
                )

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    contents = await read_upload_within_limit(file)

    # ── Load + auto-detect schema + feature-prep inline (fast — no need to queue this part) ──
    df = load_csv(contents)
    try:
        sch = schema.detect_schema(
            df, date_column=date_column, group_column=group_column, target_column=target_column
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    df = prepare_for_schema(df, sch)

    # ── Data-rows quota: counted from the actual uploaded CSV size ─────────
    await _quota_check(user, "data_rows", amount=len(df))

    job_id = training_queue.submit(user["uid"], user.get("email", "unknown"), df, sch, model_name=model_name)
    logger.info(
        f"Training job {job_id} queued for uid={user['uid']} "
        f"(date_col={sch['date_col']}, group_col={sch['group_col']}, target_col={sch['target_col']})"
    )

    return JSONResponse({
        "message": "Training job queued. Poll GET /train/status/{job_id} for progress.",
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "detected_schema": sch,
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
    model_id: str = Query(..., description="Which of your saved trained models to forecast with — see GET /models"),
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

    # ── Subscription + forecast-horizon check (doesn't touch usage yet —
    #    we don't know the exact forecasted-data-point count until the CSV
    #    is loaded below) ──────────────────────────────────────────────────
    await _quota_check(user, "forecast_points", months=months, increment=False)

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    # ── 1. Read the upload (size-capped) before touching Firestore at all ──
    # so an oversized/malicious upload gets rejected without wasting a
    # network round-trip on a model that might not even end up being used.
    contents = await read_upload_within_limit(file)

    # ── 2. Load the model the caller chose ─────────────────────────────────
    bundle = await asyncio.to_thread(download_model, user["uid"], model_id)
    if not bundle:
        raise HTTPException(
            404,
            "No saved model found with that model_id for your account. "
            "Call GET /models to see your trained models, or POST /train to create one."
        )

    # ── 3. Validate against the schema saved at training time & prepare ────
    df = load_csv(contents)
    validate_upload_against_schema(df, bundle["schema"])
    df = prepare_for_schema(df, bundle["schema"])

    # ── Data-rows quota: the forecast upload's rows count too ──────────────
    await _quota_check(user, "data_rows", amount=len(df))

    # ── Forecasted-data-points quota: one point per (series × month) ───────
    sch = bundle["schema"]
    n_series = df[sch["group_col"]].nunique() if sch["group_col"] else 1
    expected_points = n_series * months
    await _quota_check(user, "forecast_points", amount=expected_points)

    # ── 4. Forecast ───────────────────────────────────────────────────────
    preds_df = run_forecast(df, bundle, months=months)

    if preds_df.empty:
        raise HTTPException(422, "Not enough data per product to forecast")

    # ── 6. Return CSV ─────────────────────────────────────────────────────
    stream = io.StringIO()
    preds_df.to_csv(stream, index=False)
    stream.seek(0)

    metrics = bundle["metrics"]

    # ── 6.5 Save forecast to Postgres (never breaks the CSV download, but
    #      failure is now VISIBLE instead of silently swallowed — a swallowed
    #      failure here is exactly what makes "View Insights" look permanently
    #      broken and Past Forecasts look permanently empty, with no clue why.
    #      X-Forecast-Save-Error carries the real reason to the browser so it
    #      can be seen immediately (dashboard.html surfaces it), and it's
    #      logged at ERROR (not just warning) with a full traceback. ──────────
    saved_forecast_id = None
    save_error = None
    try:
        saved_forecast_id = await asyncio.to_thread(
            save_forecast_to_history,
            user["uid"],
            user.get("email", "unknown"),
            months,
            metrics,
            preds_df,
            df,
            bundle["schema"],
            model_id,
        )
    except Exception as fs_exc:
        save_error = str(fs_exc)
        logger.error(f"Forecast-history save FAILED for uid={user['uid']}: {fs_exc}", exc_info=True)

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
            "X-Forecast-Id":    saved_forecast_id or "",
            # Empty string when the save succeeded; the real exception
            # message (truncated) when it didn't, so the UI can actually
            # tell the customer their forecast wasn't saved to history
            # instead of Insights just looking mysteriously disabled.
            "X-Forecast-Save-Error": " ".join((save_error or "").split())[:200],
            # Browsers block JS (fetch) from reading response headers unless
            # the server explicitly allows it — without this, dashboard.html
            # couldn't read X-Forecast-Id to build the "View Insights" link.
            "Access-Control-Expose-Headers": "X-Forecast-Id, X-Model-Name, X-Train-RMSE, X-Val-RMSE, X-Baseline-RMSE, X-Best-Lags, X-Best-Roll, X-Forecast-Months, X-Forecast-Save-Error",
        },
    )


# =============================================================================
# 🗂️  MODELS — list / inspect / delete your saved trained models
# =============================================================================
@app.get("/models", summary="List your saved trained models (for the dashboard's model picker)")
async def get_models(user=Depends(verify_user)):
    """
    يرجّع كل الـ models المحفوظة لليوزر (الأحدث أولاً) — يستخدمها الداشبورد
    عشان يعرض قائمة الموديلات ويختار منها اليوزر أي موديل يستخدم وقت الـ forecast.
    """
    items = await asyncio.to_thread(list_models, user["uid"])
    max_models = None
    if not user.get("is_dev_bypass"):
        sub = await asyncio.to_thread(billing.get_subscription, db, user["uid"])
        if billing.is_subscription_active(sub):
            max_models = PLANS[sub["plan"]]["max_models"]
    return JSONResponse({"models": items, "count": len(items), "max_models": max_models})


@app.get("/models/{model_id}", summary="Get one saved model's metrics")
def get_model(model_id: str, user=Depends(verify_user)):
    bundle = download_model(user["uid"], model_id)
    if not bundle:
        raise HTTPException(404, "No saved model found with that model_id for your account.")

    metrics = bundle["metrics"]
    return JSONResponse({
        "id":            model_id,
        "model_name":    metrics["model_name"],
        "owner_email":   bundle.get("owner_email", "unknown"),
        "train_rmse":    round(metrics["train_rmse"],    4),
        "val_rmse":      round(metrics["val_rmse"],      4),
        "baseline_rmse": round(metrics["baseline_rmse"], 4),
        "best_lags":     bundle["lags"],
        "best_roll":     bundle["roll"],
        "schema":        bundle.get("schema"),
    })


@app.delete("/models/{model_id}", summary="Delete one of your saved trained models")
async def delete_model(model_id: str, user=Depends(verify_user)):
    """
    يحذف موديل واحد محفوظ لليوزر (بيفضي slot من الـ max_models بتاع الخطة).
    لو اليوزر عايز يستخدم الموديل ده تاني، لازم يعيد الـ /train.
    """
    deleted = await asyncio.to_thread(delete_model_cloud, user["uid"], model_id)
    if not deleted:
        raise HTTPException(404, "No model found with that model_id to delete.")
    return JSONResponse({"message": "✅ Model deleted."})


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

    Deliberately NOT gated behind an active subscription or insights_enabled —
    this is just a list of what you've already run; only opening the full
    detail (GET /forecasts/{id}) requires an Insights-capable plan.
    """
    try:
        items = forecast_store.list_forecasts(user["uid"], limit)
        return JSONResponse({"forecasts": items})
    except Exception as exc:
        logger.error(f"GET /forecasts failed for uid={user['uid']}: {exc}", exc_info=True)
        raise HTTPException(500, f"Couldn't load your forecast history: {exc}")


@app.get("/forecasts/{forecast_id}", summary="Get full data for one saved forecast (predictions + historical) — requires a plan with Insights enabled")
def get_forecast(forecast_id: str, user=Depends(verify_user)):
    """
    يرجّع forecast واحد بالتفصيل (predictions + historical snapshot) —
    ده اللي insights.html بيستخدمه لبناء الصفحة من غير الحاجة لـ sessionStorage.

    Gated behind the subscriber's plan: the Insights dashboard is a
    Growth/Scale feature (see config.PLANS[...]["insights_enabled"]) — Starter
    accounts get a clear 403 pointing them at /billing/plans instead of a
    silent failure.
    """
    if not user.get("is_dev_bypass"):
        sub = billing.get_subscription(db, user["uid"])
        if not billing.is_subscription_active(sub):
            raise HTTPException(402, "No active subscription. Choose a plan at /billing/plans to use Insights.")
        plan = PLANS[sub["plan"]]
        if not plan.get("insights_enabled"):
            raise HTTPException(
                403,
                f"The Business Insights dashboard isn't included in your '{plan['name']}' plan. "
                "Upgrade to Growth or Scale at /billing/plans to unlock it."
            )

    data = forecast_store.get_forecast(user["uid"], forecast_id)
    if not data:
        raise HTTPException(404, "Forecast not found")
    return JSONResponse(data)


@app.delete("/forecasts/{forecast_id}", summary="Delete one of your saved forecast runs")
async def delete_forecast(forecast_id: str, user=Depends(verify_user)):
    """
    يحذف forecast run واحد محفوظ لليوزر (وبيلغي وصوله لصفحة /insights بتاعته).
    ما بيأثرش على الموديل نفسه ولا على أي forecasts تانية.
    """
    deleted = await asyncio.to_thread(forecast_store.delete_forecast, user["uid"], forecast_id)
    if not deleted:
        raise HTTPException(404, "No saved forecast found with that forecast_id for your account.")
    return JSONResponse({"message": "✅ Forecast deleted."})


# =============================================================================
# 💰  SUBSCRIPTION / BILLING ENDPOINTS
# =============================================================================
# Three tiered plans (see config.PLANS: starter / growth / scale). Paying via
# Paymob activates the subscription automatically through the webhook —
# INACTIVE until you add real Paymob credentials to .env (see paymob.py).
@app.get("/billing/plans", summary="List available subscription plans")
def list_plans(user: Optional[dict] = Depends(_optional_user)):
    # trial_available is per-account (needs a logged-in user); anonymous
    # callers just see the plans with trial_available omitted.
    trial_available = None
    if user and not user.get("is_dev_bypass"):
        trial_available = not billing.has_used_trial(db, user["uid"])

    return JSONResponse({
        "plans": [
            {
                "id": plan_id,
                **plan,
                "forecast_horizon_options": forecast_horizon_options(plan),
            }
            for plan_id, plan in PLANS.items()
            # test_plan is a 25 EGP plan for exercising the Paymob checkout
            # flow without paying full price — hide it from real customers
            # by only ever showing it outside production. Remove this
            # whole entry from PLANS in config.py once testing is done.
            if plan_id != "test_plan" or not settings.is_production
        ],
        # Customer-friendly copy for the Plan Details page — one entry per
        # feature concept, filled in with each plan's own numbers client-side.
        "feature_explanations": PLAN_FEATURE_EXPLANATIONS,
        "paymob_configured": settings.paymob_configured,
        "base_currency": settings.BASE_CURRENCY,
        # Display-only currency list for the country/currency picker — every
        # plan is actually charged in base_currency regardless of which of
        # these the customer picks (see settings.display_price).
        "currencies": settings.DISPLAY_CURRENCIES,
        "trial_duration_days": billing.TRIAL_DURATION_DAYS,
        "trial_available": trial_available,
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
        "is_trial": bool(sub.get("is_trial")),
        "trial_available": not billing.has_used_trial(db, user["uid"]),
    }
    if active:
        plan = PLANS[sub["plan"]]
        try:
            usage = billing.get_usage(db, user["uid"], sub["cycle_id"])
            response["usage"] = {
                "trainings_used": usage.get("trainings_used", 0),
                "trainings_limit": plan["max_trainings"],
                "data_rows_used": usage.get("data_rows_used", 0),
                "data_rows_limit": plan["max_data_rows_per_month"],
                "forecast_points_used": usage.get("forecast_points_used", 0),
                "forecast_points_limit": plan["max_forecast_points_per_month"],
                "models_used": count_models(user["uid"]),
                "models_limit": plan["max_models"],
            }
        except Exception as exc:
            # Never let a usage-counting hiccup take down the whole billing
            # status response — insights_enabled/forecast_horizon_options
            # below (what actually gates the UI) must still come through.
            logger.error(f"billing_status usage block failed for uid={user['uid']}: {exc}")
            response["usage"] = None
        response["insights_enabled"] = plan.get("insights_enabled", False)
        response["forecast_horizon_options"] = forecast_horizon_options(plan)
    country = get_user_country(user["uid"])
    response["country"] = country
    response["currency"] = settings.DISPLAY_CURRENCIES.get(country or "", settings.DISPLAY_CURRENCIES["OTHER"])["currency"]
    return JSONResponse(response)


@app.get("/debug/firestore-check", summary="Self-test: can this account actually read/write Firestore right now?")
async def debug_firestore_check(user=Depends(verify_user)):
    """
    Writes a throwaway document to users/{uid}/_diagnostics/ping, reads it
    straight back, then deletes it — end to end, using the exact same
    Admin SDK client (`db`) as every other endpoint in this file. If
    something about this account/project/service-account is broken in a
    way that would silently break saving models or forecasts, this call
    will show the real exception right here instead of somewhere far
    upstream where it just looks like "Insights doesn't work."
    Safe to call as often as you like — cleans up after itself.
    """
    ref = db.collection("users").document(user["uid"]).collection("_diagnostics").document("ping")
    steps = {}
    try:
        test_value = {"checked_at": firestore.SERVER_TIMESTAMP, "note": "ForecastIQ Firestore self-test"}
        await asyncio.to_thread(ref.set, test_value)
        steps["write"] = "ok"

        snap = await asyncio.to_thread(ref.get)
        steps["read"] = "ok" if snap.exists else "wrote but read-back found nothing"

        await asyncio.to_thread(ref.delete)
        steps["delete"] = "ok"

        return JSONResponse({
            "ok": True,
            "uid": user["uid"],
            "steps": steps,
            "message": "Firestore read/write/delete all succeeded for this account.",
        })
    except Exception as exc:
        logger.error(f"debug_firestore_check FAILED for uid={user['uid']}: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "ok": False,
            "uid": user["uid"],
            "steps": steps,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "message": "Firestore read/write failed for this account — see 'error' above for the exact reason.",
        })


@app.get("/currencies", summary="List countries and their display currency (for the sign-up country picker)")
def get_currencies():
    """Public — used on the sign-up form before the user has an account.
    Purely for showing an approximate local-currency price; the customer
    is always actually billed in settings.BASE_CURRENCY through the one
    configured Paymob integration, regardless of the country they pick."""
    return JSONResponse({
        "base_currency": settings.BASE_CURRENCY,
        "currencies": {code: info for code, info in settings.DISPLAY_CURRENCIES.items()},
    })


class CountryUpdate(BaseModel):
    country: str  # ISO 3166-1 alpha-2 code, or "OTHER"


@app.post("/profile/country", summary="Save the account's country (sets which currency prices are displayed in)")
async def set_profile_country(body: CountryUpdate, user=Depends(verify_user)):
    code = (body.country or "").upper().strip()
    if code not in settings.DISPLAY_CURRENCIES:
        code = "OTHER"
    await asyncio.to_thread(
        lambda: db.collection("users").document(user["uid"]).set({"country": code}, merge=True)
    )
    return JSONResponse({"country": code, "currency": settings.DISPLAY_CURRENCIES[code]["currency"]})


def get_user_country(uid: str) -> Optional[str]:
    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        return None
    return doc.to_dict().get("country")


class SubscribeRequest(BaseModel):
    plan_id: str  # "starter" | "growth" | "scale"


class CheckoutRequest(BaseModel):
    plan_id: str
    # Optional override — normally we use the country already saved on the
    # account (set from the dashboard's "Country & Currency" tab). Only
    # needed if the account has no saved country yet.
    country: Optional[str] = None


@app.post(
    "/billing/start-trial",
    summary="Start a 14-day free trial of a plan (no card required, one trial per account)",
)
async def start_trial(body: SubscribeRequest, user=Depends(verify_user)):
    if user.get("is_dev_bypass"):
        raise HTTPException(400, "The developer bypass account already skips billing entirely — no trial needed.")

    if body.plan_id not in PLANS:
        raise HTTPException(400, f"Unknown plan_id. Choose one of: {', '.join(PLANS)}")

    try:
        result = await asyncio.to_thread(billing.start_trial, db, user["uid"], body.plan_id)
    except TrialAlreadyUsed as exc:
        raise HTTPException(409, str(exc))

    plan = PLANS[body.plan_id]
    logger.info(f"🎁 Trial started for uid={user['uid']}: plan={body.plan_id}, expires={result['expires_at']}")

    return JSONResponse({
        "message": f"14-day free trial of '{plan['name']}' started — no card required.",
        "plan_id": body.plan_id,
        "expires_at": result["expires_at"],
        "cycle_id": result["cycle_id"],
    })


@app.post(
    "/billing/subscribe",
    summary="Start a Paymob checkout for a subscription plan",
)
async def subscribe(body: CheckoutRequest, user=Depends(verify_user)):
    if body.plan_id not in PLANS:
        raise HTTPException(400, f"Unknown plan_id. Choose one of: {', '.join(PLANS)}")

    plan = PLANS[body.plan_id]
    email = user.get("email", "unknown@local.test")

    # Country comes from the account's saved profile (set in the dashboard's
    # "Country & Currency" tab) — no more re-picking it at checkout. `body.country`
    # is only a fallback for an account that somehow has none saved yet.
    country = (body.country or get_user_country(user["uid"]) or "OTHER").upper()
    if country not in settings.DISPLAY_CURRENCIES:
        country = "OTHER"

    amount_base = plan["price_egp"]
    display = settings.display_price(amount_base, country)

    # What we actually send to Paymob. If CHARGE_IN_DISPLAY_CURRENCY is on
    # (see config.py), the customer is charged the converted amount in
    # their own currency, through the same single PAYMOB_INTEGRATION_ID —
    # ⚠️ confirm with Paymob support that this integration accepts multiple
    # settlement currencies before relying on this in production. If it
    # doesn't, flip CHARGE_IN_DISPLAY_CURRENCY off to always charge
    # settings.BASE_CURRENCY instead (the display price still shows the
    # customer their local-currency equivalent either way).
    if settings.CHARGE_IN_DISPLAY_CURRENCY:
        charge_amount, charge_currency = display["amount"], display["currency"]
    else:
        charge_amount, charge_currency = amount_base, settings.BASE_CURRENCY

    try:
        result = await paymob.create_payment_intent(
            amount_cents=int(round(charge_amount * 100)),
            billing_email=email,
            integration_id=settings.PAYMOB_INTEGRATION_ID,
            currency=charge_currency,
            country=country,
        )
    except PaymobNotConfigured as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.error(f"Paymob request failed for uid={user['uid']} (country={country}, currency={charge_currency}): {exc}")
        raise HTTPException(502, f"Paymob request failed: {exc}")

    # Remember which plan this order is for, so the webhook (which only
    # gets Paymob's order id back, not our plan_id) knows what to
    # activate once payment succeeds.
    await asyncio.to_thread(
        lambda: db.collection("pending_orders").document(str(result["order_id"])).set({
            "uid": user["uid"],
            "plan_id": body.plan_id,
            "country": country,
            "charge_currency": charge_currency,
            "charge_amount": charge_amount,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
    )
    # Remember the chosen country on the account too, in case it came from
    # the fallback above and wasn't already saved.
    await asyncio.to_thread(
        lambda: db.collection("users").document(user["uid"]).set({"country": country}, merge=True)
    )

    return JSONResponse({
        "plan_id": body.plan_id,
        "country": country,
        "currency": charge_currency,
        "amount_charged": round(charge_amount, 2),
        "order_id": result["order_id"],
        "checkout_url": result["iframe_url"],
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