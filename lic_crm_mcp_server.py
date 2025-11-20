# lic_crm_mcp_server.py
from typing import Dict, Any, Optional

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

from db_session import SessionLocal, init_db
from db_models import CallSummary


# Ensure DB schema exists
init_db()

# Create MCP server instance
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

    Parameters (from the LLM):

    - call_id: Exotel CallSid or your internal call ID.
    - phone_number: customer phone number in E.164 or local format.
    - customer_name: optional customer name (if known).
    - interest_score: integer 0–10 where:
        0 = never buying,
        9–10 = very hot lead.
    - intent: high-level intent like: 'buy_term', 'renew', 'info_only',
              'not_interested', 'other'.
    - next_action: follow_up, whatsapp_quote, no_contact, or other.
    - raw_summary: 3–6 sentence natural language summary of the call.
    """

    db = SessionLocal()
    try:
        cs = CallSummary(
            call_id=call_id,
            phone_number=phone_number,
            customer_name=customer_name,
            intent=intent,
            interest_score=int(interest_score),
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


# Expose MCP server via SSE (HTTP) for Render / external clients
app = Starlette(
    routes=[
        # This mounts FastMCP's SSE app at the root path "/"
        Mount("/", app=mcp.sse_app()),
    ]
)
