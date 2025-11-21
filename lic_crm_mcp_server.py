from typing import Optional, Dict, Any, List

from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
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
    based on recent call summaries (heuristic).
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


# --------- Simple HTTP endpoints for testing with curl ---------

async def health(request: Request):
    return PlainTextResponse("ok")

async def test_save_http(request: Request):
    """
    Simple HTTP POST endpoint to test save_call_summary via curl.
    Expects JSON body with the same fields as the MCP tool.
    """
    body = await request.json()
    result = save_call_summary(
        call_id=body["call_id"],
        phone_number=body["phone_number"],
        customer_name=body.get("customer_name"),
        interest_score=body["interest_score"],
        intent=body["intent"],
        next_action=body.get("next_action"),
        raw_summary=body["raw_summary"],
    )
    return JSONResponse(result)


# Expose both:
# - /mcp  -> MCP SSE endpoint (for OpenAI Agents/Realtime)
# - /health and /test-save -> simple HTTP endpoints (for you)
app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/test-save", test_save_http, methods=["POST"]),
        Mount("/mcp", app=mcp.sse_app()),
    ]
)
