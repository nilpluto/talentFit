"""Shared behavior for TalentFit data models."""

from pydantic import BaseModel, ConfigDict


class TalentFitModel(BaseModel):
    """Base model with consistent string normalization."""

    model_config = ConfigDict(str_strip_whitespace=True)
