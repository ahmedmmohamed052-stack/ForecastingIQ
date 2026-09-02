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


class QuotaExceeded(Exception):
    def __init__(self, message: str):
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
        "forecasts_used": 0,
        "cycle_started_at": now,
    })

    return {"plan": plan_id, "expires_at": expires_at.isoformat(), "cycle_id": cycle_id}


def get_usage(db, uid: str, cycle_id: str) -> dict:
    doc = db.collection("users").document(uid).collection("usage").document(cycle_id).get()
    if not doc.exists:
        return {"trainings_used": 0, "forecasts_used": 0}
    return doc.to_dict()


def check_and_increment_quota(db, uid: str, kind: str) -> None:
    """
    Raises QuotaExceeded (caller turns this into HTTP 402/403) if the user
    has no active subscription, or has used up their plan's allowance for
    `kind` ("training" or "forecast") this billing cycle. Otherwise
    atomically increments the counter and returns.
    """
    sub = get_subscription(db, uid)

    if not is_subscription_active(sub):
        raise QuotaExceeded(
            "No active subscription. Choose a plan at /billing/plans and "
            "subscribe via /billing/subscribe to use this feature."
        )

    plan = PLANS[sub["plan"]]
    cycle_id = sub["cycle_id"]
    usage_ref = db.collection("users").document(uid).collection("usage").document(cycle_id)

    field = "trainings_used" if kind == "training" else "forecasts_used"
    limit_field = "max_trainings" if kind == "training" else "max_forecasts"
    limit = plan[limit_field]

    @firestore.transactional
    def _increment(transaction):
        snapshot = usage_ref.get(transaction=transaction)
        current = snapshot.to_dict().get(field, 0) if snapshot.exists else 0
        if current >= limit:
            raise QuotaExceeded(
                f"You've used all {limit} {kind} runs included in your "
                f"'{plan['name']}' plan for this billing period. It renews "
                f"on {sub['expires_at'].strftime('%Y-%m-%d')}, or upgrade "
                f"your plan at /billing/plans."
            )
        transaction.set(usage_ref, {field: current + 1}, merge=True)

    transaction = db.transaction()
    _increment(transaction)
