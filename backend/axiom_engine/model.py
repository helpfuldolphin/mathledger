from backend.repro.determinism import deterministic_uuid
from __future__ import annotations
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
import uuid

Base = declarative_base()

class Statement(Base):
    __tablename__ = 'statements'

    id = Column(String, primary_key=True, default=lambda: str(deterministic_uuid(str(content))))
    system_id = Column(String, nullable=False)  # Simplified for testing
    hash = Column(String, nullable=False, unique=True)
    content_norm = Column(Text, nullable=False)
    content = Column(Text)
    content_lean = Column(Text)
    content_latex = Column(Text)
    status = Column(String, default='unknown')
    derivation_rule = Column(String)
    derivation_depth = Column(Integer)
    created_at = Column(DateTime, nullable=False)

    # Add a property to map content_norm to normalized_text for compatibility
    @property
    def normalized_text(self):
        return self.content_norm

    @normalized_text.setter
    def normalized_text(self, value):
        self.content_norm = value
