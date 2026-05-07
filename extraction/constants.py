"""constants.py — Shared API URLs and extraction window."""

STRATZ_GRAPHQL_URL = "https://api.stratz.com/graphql"
STRATZ_API_BASE    = "https://api.stratz.com/api/v1"

WINDOW_END = 600  # capture events in [0, WINDOW_END) seconds — first 10 minutes
