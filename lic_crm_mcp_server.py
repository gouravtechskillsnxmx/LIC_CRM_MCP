from typing import Optional, Dict, Any

from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

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


app = Starlette(
    routes=[
        Mount("/", app=mcp.sse_app()),
    ]
)
