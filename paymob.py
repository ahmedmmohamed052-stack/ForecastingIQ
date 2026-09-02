"""
💳  Paymob payment gateway integration.

This module is fully wired but INACTIVE until you fill in the four
PAYMOB_* variables in your .env (see .env.example). Until then, every
function here raises a clear PaymobNotConfigured error, and the API
endpoints in main.py turn that into a friendly 503 response — so the rest
of the app (auth, train, forecast) keeps working normally without payments.

Paymob's flow (standard "Accept" API, used across Egypt/MENA) is 3 steps:
  1. Auth        -> POST /api/auth/tokens              (api key -> auth token)
  2. Order       -> POST /api/ecommerce/orders          (auth token -> order id)
  3. Payment key -> POST /api/acceptance/payment_keys   (order + amount -> payment token)
Then you redirect the user to:
  https://accept.paymob.com/api/acceptance/iframes/{IFRAME_ID}?payment_token={token}

Once you have your real credentials from https://accept.paymob.com/portal2/en/login,
drop them into .env:
    PAYMOB_API_KEY=...
    PAYMOB_INTEGRATION_ID=...
    PAYMOB_IFRAME_ID=...
    PAYMOB_HMAC_SECRET=...
No code changes needed — it activates automatically.
"""
import hashlib
import hmac as hmac_lib

import httpx

from config import settings

PAYMOB_BASE_URL = "https://accept.paymob.com/api"


class PaymobNotConfigured(Exception):
    """Raised whenever a Paymob call is attempted without credentials set."""
    pass


def _require_configured():
    if not settings.paymob_configured:
        raise PaymobNotConfigured(
            "Paymob is not configured yet. Add PAYMOB_API_KEY, "
            "PAYMOB_INTEGRATION_ID and PAYMOB_IFRAME_ID to your .env file "
            "(get them from your Paymob dashboard) to enable payments."
        )


async def get_auth_token() -> str:
    """Step 1: exchange the API key for a short-lived auth token."""
    _require_configured()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{PAYMOB_BASE_URL}/auth/tokens",
            json={"api_key": settings.PAYMOB_API_KEY},
        )
        resp.raise_for_status()
        return resp.json()["token"]


async def create_order(auth_token: str, amount_cents: int, currency: str = "EGP") -> int:
    """Step 2: register an order with Paymob, returns the Paymob order id."""
    _require_configured()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{PAYMOB_BASE_URL}/ecommerce/orders",
            json={
                "auth_token": auth_token,
                "delivery_needed": False,
                "amount_cents": amount_cents,
                "currency": currency,
                "items": [],
            },
        )
        resp.raise_for_status()
        return resp.json()["id"]


async def get_payment_key(
    auth_token: str,
    order_id: int,
    amount_cents: int,
    billing_email: str,
    currency: str = "EGP",
) -> str:
    """Step 3: request the payment token used to open the iframe."""
    _require_configured()
    billing_data = {
        "email": billing_email or "customer@example.com",
        "first_name": "N/A",
        "last_name": "N/A",
        "phone_number": "N/A",
        "apartment": "N/A", "floor": "N/A", "street": "N/A",
        "building": "N/A", "shipping_method": "N/A", "postal_code": "N/A",
        "city": "N/A", "country": "N/A", "state": "N/A",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{PAYMOB_BASE_URL}/acceptance/payment_keys",
            json={
                "auth_token": auth_token,
                "amount_cents": amount_cents,
                "expiration": 3600,
                "order_id": order_id,
                "billing_data": billing_data,
                "currency": currency,
                "integration_id": settings.PAYMOB_INTEGRATION_ID,
            },
        )
        resp.raise_for_status()
        return resp.json()["token"]


async def create_payment_intent(amount_cents: int, billing_email: str, currency: str = "EGP") -> dict:
    """
    One-shot helper: runs all 3 Paymob steps and returns the iframe URL
    the frontend should redirect the user to.
    """
    _require_configured()
    auth_token = await get_auth_token()
    order_id = await create_order(auth_token, amount_cents, currency)
    payment_token = await get_payment_key(auth_token, order_id, amount_cents, billing_email, currency)
    iframe_url = (
        f"https://accept.paymob.com/api/acceptance/iframes/"
        f"{settings.PAYMOB_IFRAME_ID}?payment_token={payment_token}"
    )
    return {"order_id": order_id, "payment_token": payment_token, "iframe_url": iframe_url}


def verify_hmac(received_hmac: str, data: dict) -> bool:
    """
    Verifies the HMAC signature Paymob sends on webhook callbacks, so you
    can trust that a "transaction processed" notification really came from
    Paymob and wasn't spoofed. Field order below is Paymob's documented
    concatenation order for the `TRANSACTION` webhook — do not reorder.
    """
    _require_configured()
    ordered_keys = [
        "amount_cents", "created_at", "currency", "error_occured",
        "has_parent_transaction", "id", "integration_id", "is_3d_secure",
        "is_auth", "is_capture", "is_refunded", "is_standalone_payment",
        "is_voided", "order", "owner", "pending", "source_data_pan",
        "source_data_sub_type", "source_data_type", "success",
    ]
    concatenated = "".join(str(data.get(k, "")) for k in ordered_keys)
    computed = hmac_lib.new(
        settings.PAYMOB_HMAC_SECRET.encode(),
        concatenated.encode(),
        hashlib.sha512,
    ).hexdigest()
    return hmac_lib.compare_digest(computed, received_hmac)
