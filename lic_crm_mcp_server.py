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

import os

# --- ML imports (real model via scikit-learn) ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
except ImportError:
    # If scikit-learn / joblib are not installed, these stay None and
    # the ML tools will return a clear error message.
    TfidfVectorizer = None
    LogisticRegression = None
    joblib = None

# Where to store the trained model inside the container
ML_MODEL_PATH = os.getenv(
    "LIC_CRM_ML_MODEL_PATH",
    "/tmp/lic_crm_promising_calls_model.joblib",
)

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


# ---------------- ML helpers: training + ranking ---------------- #

def _check_ml_dependencies() -> Optional[str]:
    """
    Return an error message if ML dependencies are missing, else None.
    """
    if TfidfVectorizer is None or LogisticRegression is None or joblib is None:
        return (
            "scikit-learn and joblib are required but not installed. "
            "Add 'scikit-learn' and 'joblib' to requirements.txt and redeploy."
        )
    return None


def _load_labeled_calls_for_training() -> List[CallSummary]:
    """
    Load calls that have a known 'purchased' label for supervised training.
    We use purchased=True as positive class (1), purchased=False as negative (0).
    """
    db = SessionLocal()
    try:
        # Only rows where purchased is not NULL
        rows: List[CallSummary] = (
            db.query(CallSummary)
            .filter(CallSummary.purchased.isnot(None))  # requires purchased column in model
            .order_by(CallSummary.call_timestamp)
            .all()
        )
        return rows
    finally:
        db.close()


def _train_promising_calls_model_impl() -> Dict[str, Any]:
    """
    Train a logistic regression model to predict purchase likelihood from
    (intent + raw_summary). Saves the model to ML_MODEL_PATH.

    This is 'real ML' (supervised learning) and will be as good as the labels
    in your call_summaries.purchased column.
    """
    dep_err = _check_ml_dependencies()
    if dep_err:
        return {"status": "error", "message": dep_err}

    rows = _load_labeled_calls_for_training()
    if not rows:
        return {
            "status": "error",
            "message": (
                "No labeled calls found for training. "
                "You must set 'purchased' = true/false on some CallSummary rows."
            ),
        }

    # Build texts and labels
    texts: List[str] = []
    labels: List[int] = []

    for cs in rows:
        # Combine intent + raw_summary as the text feature
        text = f"{cs.intent or ''}. {cs.raw_summary or ''}"
        texts.append(text)
        # Positive class = purchased=True
        labels.append(1 if cs.purchased else 0)

    n_samples = len(texts)
    n_pos = sum(labels)
    n_neg = n_samples - n_pos

    if n_pos < 2 or n_neg < 2:
        return {
            "status": "error",
            "message": (
                f"Not enough labeled data to train. "
                f"Need at least 2 positives and 2 negatives; have {n_pos} positives, {n_neg} negatives."
            ),
            "total_samples": n_samples,
            "positives": n_pos,
            "negatives": n_neg,
        }

    # Vectorize text and train logistic regression
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # handle class imbalance if positives are fewer
    )
    clf.fit(X, labels)

    # Persist model to disk
    model_obj = {"vectorizer": vectorizer, "clf": clf}
    joblib.dump(model_obj, ML_MODEL_PATH)

    return {
        "status": "ok",
        "message": "Trained logistic regression model for promising calls.",
        "model_path": ML_MODEL_PATH,
        "total_samples": n_samples,
        "positives": n_pos,
        "negatives": n_neg,
    }


def _rank_promising_calls_ml_impl(limit: int = 10) -> Dict[str, Any]:
    """
    Use the trained ML model (if available) to rank calls by purchase probability.

    Returns:
      - calls: list of call dicts with 'ml_score' in [0,1] (higher = more promising)
    """
    dep_err = _check_ml_dependencies()
    if dep_err:
        return {"status": "error", "message": dep_err}

    if not os.path.exists(ML_MODEL_PATH):
        return {
            "status": "error",
            "message": (
                f"ML model file not found at {ML_MODEL_PATH}. "
                "Call 'train_promising_calls_model' first."
            ),
        }

    model_obj = joblib.load(ML_MODEL_PATH)
    vectorizer = model_obj["vectorizer"]
    clf = model_obj["clf"]

    db = SessionLocal()
    try:
        # Fetch all calls (or you can filter to not purchased, recent, etc.)
        rows: List[CallSummary] = (
            db.query(CallSummary)
            .order_by(desc(CallSummary.call_timestamp))
            .all()
        )

        if not rows:
            return {"status": "ok", "limit": limit, "count": 0, "calls": []}

        texts: List[str] = []
        for cs in rows:
            text = f"{cs.intent or ''}. {cs.raw_summary or ''}"
            texts.append(text)

        X = vectorizer.transform(texts)
        # predict_proba gives [P(class=0), P(class=1)], we take P(class=1)
        probs = clf.predict_proba(X)[:, 1]

        scored_calls: List[Dict[str, Any]] = []
        for cs, p in zip(rows, probs):
            scored_calls.append(
                {
                    "id": cs.id,
                    "call_id": cs.call_id,
                    "phone_number": cs.phone_number,
                    "customer_name": cs.customer_name,
                    "call_timestamp": cs.call_timestamp.isoformat() if cs.call_timestamp else None,
                    "intent": cs.intent,
                    "interest_score": cs.interest_score,
                    "next_action": cs.next_action,
                    "purchased": cs.purchased,
                    "purchase_date": cs.purchase_date.isoformat() if cs.purchase_date else None,
                    "raw_summary": cs.raw_summary,
                    "ml_score": float(round(float(p), 6)),  # probability of purchase
                }
            )

        # Sort by ml_score DESC (most promising first), then recency
        scored_calls.sort(
            key=lambda r: (r["ml_score"], r["call_timestamp"] or ""),
            reverse=True,
        )

        top = scored_calls[:limit]

        return {
            "status": "ok",
            "limit": limit,
            "count": len(top),
            "calls": top,
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
    MCP tool wrapper for heuristic scoring of customers (non-ML).
    """
    return _score_customers_impl(limit=limit)


@mcp.tool()
def train_promising_calls_model() -> Dict[str, Any]:
    """
    Train a logistic regression model from historical call_summaries
    where 'purchased' is known (True/False).

    Use this before calling 'rank_promising_calls_ml'.
    """
    return _train_promising_calls_model_impl()


@mcp.tool()
def rank_promising_calls_ml(limit: int = 10) -> Dict[str, Any]:
    """
    Use the trained ML model to return the top N most promising calls.

    'ml_score' is the predicted probability that the call leads to a purchase.
    """
    return _rank_promising_calls_ml_impl(limit=limit)


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
