import os
import secrets


def authorized(api_key):
    """True if LIVENESS_API_KEY is unset (auth disabled) or api_key matches it."""
    expected = os.environ.get("LIVENESS_API_KEY")
    return not expected or secrets.compare_digest(api_key or "", expected)
