from datetime import datetime, timezone, timedelta

RECENCY_MONTHS = 24
RECENCY_CUTOFF = datetime.now(timezone.utc).date() - timedelta(days=30 * RECENCY_MONTHS)
COMMUNITY_ANCIENT_CUTOFF = datetime(2023, 1, 1).date()
