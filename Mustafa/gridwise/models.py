from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from .database import Base


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    total_grid_kwh: Mapped[float | None] = mapped_column(Float)
    total_cost_bdt: Mapped[float | None] = mapped_column(Float)
    peak_grid_kwh: Mapped[float | None] = mapped_column(Float)
    plan_summary: Mapped[str | None] = mapped_column(Text)
    request_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    response_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    battery: Mapped[Battery | None] = relationship(back_populates="run", uselist=False, cascade="all, delete-orphan")
    hours: Mapped[list[ScenarioHour]] = relationship(back_populates="run", cascade="all, delete-orphan")
    notes: Mapped[list[OperatorNote]] = relationship(back_populates="run", cascade="all, delete-orphan")
    directives: Mapped[list[Directive]] = relationship(back_populates="run", cascade="all, delete-orphan")
    plans: Mapped[list[HourlyPlan]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Battery(Base):
    __tablename__ = "batteries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("optimization_runs.id", ondelete="CASCADE"), nullable=False, unique=True)
    capacity_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    initial_energy_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    minimum_energy_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    max_charge_kwh_per_hour: Mapped[float] = mapped_column(Float, nullable=False)
    max_discharge_kwh_per_hour: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped[OptimizationRun] = relationship(back_populates="battery")


class ScenarioHour(Base):
    __tablename__ = "scenario_hours"
    __table_args__ = (UniqueConstraint("run_id", "hour"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("optimization_runs.id", ondelete="CASCADE"), nullable=False)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    demand_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    solar_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    tariff_bdt_per_kwh: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped[OptimizationRun] = relationship(back_populates="hours")


class OperatorNote(Base):
    __tablename__ = "operator_notes"
    __table_args__ = (UniqueConstraint("run_id", "note_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("optimization_runs.id", ondelete="CASCADE"), nullable=False)
    note_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    run: Mapped[OptimizationRun] = relationship(back_populates="notes")


class Directive(Base):
    __tablename__ = "directives"
    __table_args__ = (UniqueConstraint("run_id", "note_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("optimization_runs.id", ondelete="CASCADE"), nullable=False)
    note_index: Mapped[int] = mapped_column(Integer, nullable=False)
    applies: Mapped[bool] = mapped_column(Boolean, nullable=False)
    directive_type: Mapped[str] = mapped_column(String(64), nullable=False)
    structured_adjustment: Mapped[dict | None] = mapped_column(JSON)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)

    run: Mapped[OptimizationRun] = relationship(back_populates="directives")


class HourlyPlan(Base):
    __tablename__ = "hourly_plans"
    __table_args__ = (UniqueConstraint("run_id", "hour"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("optimization_runs.id", ondelete="CASCADE"), nullable=False)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    grid_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    solar_used_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    battery_action: Mapped[str] = mapped_column(String(16), nullable=False)
    battery_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    battery_energy_after_kwh: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped[OptimizationRun] = relationship(back_populates="plans")