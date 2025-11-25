from typing import Optional, Dict, Any, List

from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from sqlalchemy import func, desc

from db_session import SessionLocal, init_db
from db_models import CallSummary
from datetime import datetime, timezone

# Ensure DB tables exist
init_db()

# Create MCP server
mcp = FastMCP("lic-crm-mcp")


# ---------------- Core implementation functions (plain Python) ---------------- #

def _save_call_summary_impl(
    call_id: str,
    phone_number: str,
    customer_name: Optional[str],
    interest_score: int,
    intent: str,
    next_action: Optional[str],
    raw_summary: str,
) -> Dict[str, Any]:
    """
    Actual implementation that writes a call summary into the database.
    This is called both by:
      - the MCP tool wrapper, and
      - the /test-save HTTP endpoint (for debug).
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
            # important for ws_server.py: always stamp when the summary is saved
            call_timestamp=datetime.now(timezone.utc),
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


def _score_customers_impl(limit: int = 5) -> Dict[str, Any]:
    """
    Heuristic scoring implementation:
    - For each phone_number, compute max(interest_score) and last call timestamp.
    - Sort by max_interest desc, then last_call desc.
    """

    db = SessionLocal()
    try:
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


# ---------------- MCP tool wrappers (what FastMCP exposes) ---------------- #

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
    MCP tool wrapper that delegates to the core implementation.

    This is what ws_server.py should call after the call is summarized.
    """
    return _save_call_summary_impl(
        call_id=call_id,
        phone_number=phone_number,
        customer_name=customer_name,
        interest_score=interest_score,
        intent=intent,
        next_action=next_action,
        raw_summary=raw_summary,
    )


@mcp.tool()
def score_customers(limit: int = 5) -> Dict[str, Any]:
    """
    MCP tool wrapper for scoring customers.
    """
    return _score_customers_impl(limit=limit)


# ---------------- Simple HTTP endpoints for manual testing ---------------- #

async def health(request: Request):
    return PlainTextResponse("ok")


async def test_save_http(request: Request):
    """
    Simple HTTP POST endpoint to test saving a call summary via curl.
    Expects JSON body with the same fields as the MCP tool.
    """
    body = await request.json()

    # Append timestamp to base_id so /test-save never collides on call_id
    base_id = body.get("call_id", "debug_call")
    call_id = f"{base_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    result = _save_call_summary_impl(
        call_id=call_id,
        phone_number=body["phone_number"],
        customer_name=body.get("customer_name"),
        interest_score=body["interest_score"],
        intent=body["intent"],
        next_action=body.get("next_action"),
        raw_summary=body["raw_summary"],
    )
    return JSONResponse(result)


# Expose:
# - /mcp       -> MCP SSE endpoint (for OpenAI Agents/Realtime)
# - /health    -> health check
# - /test-save -> simple HTTP JSON endpoint for you to test with curl
app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/test-save", test_save_http, methods=["POST"]),
        Mount("/mcp", app=mcp.sse_app()),
    ]
)
