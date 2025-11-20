from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CallSummary(Base):
    """
    Stores one summarised LIC sales call.
    """
    __tablename__ = "call_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)

    call_id = Column(String(64), unique=True, index=True, nullable=False)
    phone_number = Column(String(20), index=True, nullable=False)
    customer_name = Column(String(100))

    call_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    intent = Column(String(50), nullable=False)          # e.g. buy_term, renew, info_only
    interest_score = Column(Integer, nullable=False)     # 0–10
    next_action = Column(String(50))                     # e.g. follow_up, whatsapp_quote

    raw_summary = Column(Text, nullable=False)           # natural language summary

    purchased = Column(Boolean, default=None)
    purchase_date = Column(DateTime, nullable=True)
