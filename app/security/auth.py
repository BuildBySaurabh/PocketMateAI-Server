import hmac
from typing import Optional


def token_is_valid(authorization_header: Optional[str], expected_token: str) -> bool:
    """Checks `Authorization: Bearer <token>` in constant time. An unset server token never matches."""
    if not expected_token or not authorization_header:
        return False
    scheme, _, supplied = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not supplied:
        return False
    return hmac.compare_digest(supplied.strip().encode(), expected_token.encode())
