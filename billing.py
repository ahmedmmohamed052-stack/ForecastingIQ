"""
💰  Subscriptions & usage quotas.

Three plans (see config.PLANS): forecast_3mo / forecast_6mo / forecast_12mo
— priced by how far ahead a plan lets you forecast, not by subscription
access length (every plan bills on the same 30-day cycle). A user's
subscription lives in Firestore at users/{uid} (fields: plan, status,
expires_at, started_at). Usage counters live at
users/{uid}/usage/{plan_cycle_id} and reset automatically whenever a new
billing period starts (renewal or plan change).

Flow:
  1. Frontend calls POST /billing/subscribe with a plan id
  2. We create a Paymob payment intent for that plan's price
  3. User pays via the Paymob iframe
  4. Paymob calls POST /payment/webhook -> activate_subscription() runs
  5. From then on, /train and /forecast call check_and_increment_quota()
     before doing any real work
"""
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from firebase_admin import firestore

from config import PLANS

# ── Free trial ────────────────────────────────────────────────────────────
# 14 days, full access to whichever plan the user picks, no card required.
# One trial per account EVER — not per plan. Once used (on any plan), the
# user doc's "trial_used" flag is set permanently and start_trial() refuses
# every later attempt, even for a different plan_id.
TRIAL_DURATION_DAYS = 14


class QuotaExceeded(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class TrialAlreadyUsed(Exception):
    def __init__(self, message: str = "You've already used your free trial. Choose a plan at /billing/plans to subscribe."):
        self.message = message
        super().__init__(message)


def get_subscription(db, uid: str) -> dict:
    """Returns the user's current subscription doc, or a default 'no plan' shape."""
    doc = db.collection("users").document(uid).get()
    data = doc.to_dict() if doc.exists else {}
    return {
        "plan": data.get("plan"),
        "status": data.get("status", "inactive"),
        "started_at": data.get("started_at"),
        "expires_at": data.get("expires_at"),
        "cycle_id": data.get("cycle_id"),
    }


def is_subscription_active(sub: dict) -> bool:
    if sub.get("status") != "active" or not sub.get("plan"):
        return False
    expires_at = sub.get("expires_at")
    if expires_at is None:
        return False
    # Firestore timestamps come back as timezone-aware datetimes already.
    return expires_at > datetime.now(timezone.utc)


def activate_subscription(db, uid: str, plan_id: str, order_id: str = None) -> dict:
    """
    Called from the Paymob webhook once a payment succeeds. Starts a brand
    new billing cycle (which also resets usage quotas) regardless of
    whether the user had a previous plan.
    """
    if plan_id not in PLANS:
        raise ValueError(f"Unknown plan_id: {plan_id}")

    plan = PLANS[plan_id]
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=plan["duration_days"])
    cycle_id = uuid.uuid4().hex[:12]

    db.collection("users").document(uid).set({
        "plan": plan_id,
        "status": "active",
        "started_at": now,
        "expires_at": expires_at,
        "cycle_id": cycle_id,
        "last_paymob_order_id": order_id,
    }, merge=True)

    # Fresh usage counters for the new cycle.
    db.collection("users").document(uid).collection("usage").document(cycle_id).set({
        "trainings_used": 0,
        "data_rows_used": 0,
        "forecast_points_used": 0,
        "cycle_started_at": now,
    })

    return {"plan": plan_id, "expires_at": expires_at.isoformat(), "cycle_id": cycle_id}


def has_used_trial(db, uid: str) -> bool:
    """Global, cross-plan check — True once the user has ever started a trial."""
    doc = db.collection("users").document(uid).get()
    data = doc.to_dict() if doc.exists else {}
    return bool(data.get("trial_used"))


def start_trial(db, uid: str, plan_id: str) -> dict:
    """
    Activates `plan_id` for TRIAL_DURATION_DAYS at no charge, no card
    required. Uses the same Firestore transaction pattern as
    activate_subscription() but is gated by a permanent, account-wide
    "trial_used" flag: once any plan's trial has been started, every
    later call — for this plan or any other — raises TrialAlreadyUsed.
    """
    if plan_id not in PLANS:
        raise ValueError(f"Unknown plan_id: {plan_id}")

    user_ref = db.collection("users").document(uid)

    @firestore.transactional
    def _start(transaction):
        snapshot = user_ref.get(transaction=transaction)
        data = snapshot.to_dict() if snapshot.exists else {}
        if data.get("trial_used"):
            raise TrialAlreadyUsed()

        plan = PLANS[plan_id]
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=TRIAL_DURATION_DAYS)
        cycle_id = uuid.uuid4().hex[:12]

        transaction.set(user_ref, {
            "plan": plan_id,
            "status": "active",
            "started_at": now,
            "expires_at": expires_at,
            "cycle_id": cycle_id,
            "is_trial": True,
            "trial_used": True,
            "trial_plan_id": plan_id,
            "trial_started_at": now,
        }, merge=True)

        usage_ref = user_ref.collection("usage").document(cycle_id)
        transaction.set(usage_ref, {
            "trainings_used": 0,
            "data_rows_used": 0,
            "forecast_points_used": 0,
            "cycle_started_at": now,
        })

        return {"plan": plan_id, "expires_at": expires_at.isoformat(), "cycle_id": cycle_id}

    transaction = db.transaction()
    return _start(transaction)


def get_usage(db, uid: str, cycle_id: str) -> dict:
    doc = db.collection("users").document(uid).collection("usage").document(cycle_id).get()
    if not doc.exists:
        return {"trainings_used": 0, "data_rows_used": 0, "forecast_points_used": 0}
    data = doc.to_dict()
    data.setdefault("trainings_used", 0)
    data.setdefault("data_rows_used", 0)
    data.setdefault("forecast_points_used", 0)
    return data


# kind -> (usage field name, plan limit field name, human label)
QUOTA_KINDS = {
    "training":        ("trainings_used",       "max_trainings",                  "training run"),
    "data_rows":       ("data_rows_used",        "max_data_rows_per_month",        "data row"),
    "forecast_points": ("forecast_points_used",  "max_forecast_points_per_month",  "forecasted data point"),
}


def check_and_increment_quota(db, uid: str, kind: str, amount: int = 1) -> None:
    """
    Raises QuotaExceeded (caller turns this into HTTP 402/403) if the user
    has no active subscription, or `amount` more of `kind`
    ("training" | "data_rows" | "forecast_points") would exceed their
    plan's monthly allowance. Otherwise atomically increments the counter
    by `amount` and returns.
    """
    sub = get_subscription(db, uid)

    if not is_subscription_active(sub):
        raise QuotaExceeded(
            "No active subscription. Choose a plan at /billing/plans and "
            "subscribe via /billing/subscribe to use this feature."
        )

    plan = PLANS[sub["plan"]]
    field, limit_field, label = QUOTA_KINDS[kind]
    limit = plan[limit_field]
    cycle_id = sub["cycle_id"]
    usage_ref = db.collection("users").document(uid).collection("usage").document(cycle_id)

    @firestore.transactional
    def _increment(transaction):
        snapshot = usage_ref.get(transaction=transaction)
        current = snapshot.to_dict().get(field, 0) if snapshot.exists else 0
        if current + amount > limit:
            remaining = max(0, limit - current)
            raise QuotaExceeded(
                f"This would use {amount} {label}s, but your '{plan['name']}' plan "
                f"only has {remaining} left this billing period (limit: {limit}). "
                f"It renews on {sub['expires_at'].strftime('%Y-%m-%d')}, or upgrade "
                f"your plan at /billing/plans."
            )
        transaction.set(usage_ref, {field: current + amount}, merge=True)

    transaction = db.transaction()
    _increment(transaction)
