"""
📋  Monitoring & error tracking.

Replaces scattered print() calls with real logging:
  - Console output (so `docker logs` / your host's log viewer works)
  - A rotating local file (logs/forecastiq.log) so history survives even
    if you're not tailing the console when something breaks
  - Optional Sentry integration: set SENTRY_DSN in .env (free tier at
    sentry.io is plenty to start) to get real-time alerts + stack traces
    for unhandled exceptions, instead of finding out from a customer email
    and then SSHing in to grep a log file.

Call setup_logging() once, at import time in main.py, before anything else
logs. Then anywhere else in the app: `logger = logging.getLogger("forecastiq")`.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from config import settings


def setup_logging():
    logger = logging.getLogger("forecastiq")
    if logger.handlers:
        return logger  # already configured (e.g. re-imported under reload)

    logger.setLevel(settings.LOG_LEVEL)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler(
        "logs/forecastiq.log", maxBytes=5_000_000, backupCount=5
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if settings.SENTRY_DSN:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.logging import LoggingIntegration

            sentry_logging = LoggingIntegration(
                level=logging.INFO,        # breadcrumbs from INFO and up
                event_level=logging.ERROR,  # send an event to Sentry on ERROR and up
            )
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                environment=settings.ENVIRONMENT,
                integrations=[FastApiIntegration(), sentry_logging],
                # Sentry normally auto-scans every installed package and
                # tries to hook into anything it recognizes (langchain,
                # torch, redis, etc.) even if this app never imports them
                # itself. On some machines that auto-scan imports a broken
                # or conflicting package (e.g. a corrupt torch DLL from an
                # unrelated project on the same Python install) and crashes
                # the whole app before it can even start — just for
                # "monitoring", which defeats the purpose. We only want the
                # two integrations we explicitly listed above.
                auto_enabling_integrations=False,
                traces_sample_rate=0.1,
            )
            logger.info("Sentry error tracking enabled")
        except ImportError:
            logger.warning(
                "SENTRY_DSN is set but the 'sentry-sdk' package isn't "
                "installed — run `pip install sentry-sdk` to enable it."
            )
        except Exception as exc:
            # Monitoring setup must never be able to take the whole app
            # down. If Sentry fails to initialize for any reason, log it
            # and keep going with console + file logging only.
            logger.warning(f"Sentry failed to initialize (continuing without it): {exc}")
    else:
        logger.info(
            "SENTRY_DSN not set — errors will only go to console + "
            "logs/forecastiq.log. Set SENTRY_DSN in .env for real-time "
            "error alerts (see https://sentry.io)."
        )

    return logger