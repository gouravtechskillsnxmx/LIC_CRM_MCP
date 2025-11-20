from typing import Optional, Dict, Any, List

from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount
from sqlalchemy import func, desc

from db_session import SessionLocal, init_db
from db_models import CallSummary

# Ensure DB tables exist
init_db()

# Create MCP server
mcp = FastMCP("lic-crm-mcp")


@mcp.tool()
def save_call_summary(
    call_id: str,
    phone_number: str,
    customer_name: Optional[str],
    interest_score: int,
    intent: str,
    next_action: Optional[str],
    raw_summary: str,
) -> Dict[str, Any]:
    """
    Save a structured summary of an LIC sales call into the database.

    Parameters:
      call_id        - Exotel CallSid or your own call ID
      phone_number   - customer's phone number (string)
      customer_name  - optional customer name
      interest_score - integer 0–10 (0 = never buying, 9–10 = very hot lead)
      intent         - e.g. 'buy_term', 'renew', 'info_only', 'not_interested', 'other'
      next_action    - e.g. 'follow_up', 'whatsapp_quote', 'no_contact', 'other'
      raw_summary    - 3–6 sentence natural language summary of the call
    """

    db = SessionLocal()
    try:
        cs = CallSummary(
            call_id=call_id,
            phone_number=phone_number,
            customer_name=customer_name,
            interest_score=int(interest_score),
            intent=intent,
            next_action=next_action,
            raw_summary=raw_summary,
        )
        db.add(cs)
        db.commit()
        db.refresh(cs)

        return {
            "status": "ok",
            "id": cs.id,
            "call_id": call_id,
            "phone_number": phone_number,
        }
    finally:
        db.close()


@mcp.tool()
def score_customers(limit: int = 5) -> Dict[str, Any]:
    """
    Return top N customers who look most likely to buy,
    based on recent call summaries.

    Initial heuristic (no ML yet):
    - For each phone_number, compute max(interest_score) and last call timestamp.
    - Sort customers by max interest_score (desc), then by last_call (desc).
    - Return up to `limit` customers.
    """

    db = SessionLocal()
    try:
        # Subquery: aggregate per phone_number
        subq = (
            db.query(
                CallSummary.phone_number.label("phone_number"),
                func.max(CallSummary.interest_score).label("max_interest"),
                func.max(CallSummary.call_timestamp).label("last_call"),
            )
            .group_by(CallSummary.phone_number)
            .subquery()
        )

        rows: List[Any] = (
            db.query(
                subq.c.phone_number,
                subq.c.max_interest,
                subq.c.last_call,
            )
            .order_by(desc(subq.c.max_interest), desc(subq.c.last_call))
            .limit(limit)
            .all()
        )

        customers = []
        for phone, max_interest, last_call in rows:
            customers.append(
                {
                    "phone_number": phone,
                    "max_interest_score": int(max_interest) if max_interest is not None else None,
                    "last_call_timestamp": last_call.isoformat() if last_call else None,
                }
            )

        return {
            "status": "ok",
            "limit": limit,
            "customers": customers,
        }
    finally:
        db.close()


# Expose MCP server via SSE for Realtime/Agents
app = Starlette(
    routes=[
        # This mounts FastMCP's SSE app at the root path "/"
        Mount("/", app=mcp.sse_app()),
    ]
)
