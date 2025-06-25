from sqlalchemy import Column, String, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from typing import Optional, List, Dict, Any
import os
from pathlib import Path

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SQLITE_DB = os.path.join(BASE_DIR, 'data', 'citations.db')

# Ensure the data directory exists
os.makedirs(os.path.dirname(SQLITE_DB), exist_ok=True)

# Create SQLAlchemy engine and session
SQLALCHEMY_DATABASE_URL = f"sqlite:///{SQLITE_DB}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

Base = declarative_base()

class Citation(Base):
    __tablename__ = "citations"
    
    id = Column(String, primary_key=True, index=True)  # Using the document ID as primary key
    source_doc_id = Column(String, index=True)  # Original document ID
    citation_count = Column(Integer, default=0)  # Number of times cited
    title = Column(String, nullable=True)  # Document title for reference
    
    @classmethod
    def get_db(cls):
        """Get a database session"""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

# Create tables
Base.metadata.create_all(bind=engine)
