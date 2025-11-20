# db_session.py
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_models import Base


# On Render, set this env var to your Postgres connection string.
# Example for local dev:
# export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/lic_bot"
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var must be set")


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
