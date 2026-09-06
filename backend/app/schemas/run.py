from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TOPOLOGY_TYPES = [
    "2D_Mesh",
    "3D_Mesh",
    "2_5D_Mesh",
    "3_5D_Mesh",
    "2_5D_Mesh_Scaled",
    "3_5D_Mesh_Scaled",
]


class HisimConfig(BaseModel):
    """Every knob the HISIM analytical engine reads out of config.py."""

    # Sweep values are assigned after the base config is parsed.  Keep the
    # same bounds/type guarantees for those assignments as for POST /runs;
    # otherwise a typo can be written into config.py and only fail minutes
    # later inside the analytical engine.
    model_config = ConfigDict(validate_assignment=True)

    # --- workload -------------------------------------------------------- #
    aimodel: str = "gpt2"
    parse_mlir_output: bool = False

    # --- run mode -------------------------------------------------------- #
    debug: bool = False
    create_default_files: bool = True
    default_files_generic: bool = True
    type_default_files: str = "2D_Mesh"
    set_suff_banks: bool = True

    # --- package / integration ------------------------------------------- #
    stack_count: int = Field(default=4, ge=1, le=256)
    chip_count: int = Field(default=1, ge=1, le=64)
    tile_count_dict: dict[str, int] = Field(
        default_factory=lambda: {"SA": 2, "CPU": 2, "Mem_I": 1, "Mem_W": 1, "Mem_O": 1}
    )

    # --- systolic array / compute ---------------------------------------- #
    sa_size_x: int = Field(default=16, ge=1, le=1024)
    sa_size_y: int = Field(default=16, ge=1, le=1024)
    n_sa: int = Field(default=2, ge=1, le=256)
    precision_bits: int = Field(default=8, ge=1, le=64)
    clock_hz: float = Field(default=1e9, gt=0)

    # --- memory ---------------------------------------------------------- #
    n_bank: int = Field(default=1, ge=1, le=256)
    n_word: int = Field(default=1024, ge=1)
    n_bit: int = Field(default=320, ge=1)
    # Mem_LUT.csv currently contains calibrated rows only for CM=4.  Exposing
    # other values makes an otherwise valid run fail deep inside the engine.
    col_mux: Literal[4] = 4

    # --- interconnect ----------------------------------------------------- #
    n_2d_links_per_tile: int = Field(default=80, ge=1, le=4096)
    n_3d_links_per_tile: int = Field(default=80, ge=1, le=4096)
    n_2_5d_channels_per_chiplet_edge: int = Field(default=1, ge=1, le=256)

    # --- escape hatch ------------------------------------------------------ #
    extra_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type_default_files")
    @classmethod
    def _valid_topology(cls, value: str) -> str:
        if value not in TOPOLOGY_TYPES:
            raise ValueError(f"Unknown topology template: {value!r}")
        return value

    def to_config_values(self) -> dict[str, Any]:
        """Map GUI field names onto the identifiers used inside config.py."""
        values: dict[str, Any] = {
            "DEBUG": self.debug,
            "CREATE_DEFAULT_FILES": self.create_default_files,
            "DEFAULT_FILES_GENERIC": self.default_files_generic,
            "TYPE_DEFAULT_FILES": self.type_default_files,
            "SET_SUFF_BANKS": self.set_suff_banks,
            "stack_count": self.stack_count,
            "chip_count": self.chip_count,
            "tile_count_dict": self.tile_count_dict,
            "aimodel": self.aimodel,
            "parse_mlir_output": self.parse_mlir_output,
            "def_SA_size_x": self.sa_size_x,
            "def_SA_size_y": self.sa_size_y,
            "def_n_SA": self.n_sa,
            "def_prec": self.precision_bits,
            "def_clk_hz": self.clock_hz,
            "def_Nbank": self.n_bank,
            "def_NW": self.n_word,
            "def_NB": self.n_bit,
            "def_CM": self.col_mux,
            "def_n_2d_links_per_tile": self.n_2d_links_per_tile,
            "def_n_3d_links_per_tile": self.n_3d_links_per_tile,
            "def_n_2_5d_channels_per_chiplet_edge": self.n_2_5d_channels_per_chiplet_edge,
        }
        values.update(self.extra_parameters)
        return values


class RunCreate(BaseModel):
    name: str | None = None
    config: HisimConfig


class SweepCreate(BaseModel):
    """Launch a one-knob sweep for the model selected in ``config``."""

    name: str | None = None
    config: HisimConfig
    knob: str
    knob_label: str | None = None
    values: list[Any] = Field(min_length=1, max_length=24)
    # Explicit opt-in protects older clients from unexpectedly multiplying a
    # sweep after the UI gained model comparison support.
    compare_models: bool = False
    models: list[str] = Field(default_factory=list, max_length=12)


RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class RunRecord(BaseModel):
    id: str
    name: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    config: HisimConfig
    return_code: int | None = None
    error: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_s: float | None = None

    # sweep bookkeeping — one group per launched batch
    group: str | None = None
    group_label: str | None = None
    sweep_knob: str | None = None
    sweep_knob_label: str | None = None
    sweep_value: str | None = None
    sweep_model: str | None = None

    @field_validator("created_at", "updated_at", mode="after")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        """Records written before the GUI stored tz-aware stamps come back naive.

        Mixing naive and aware datetimes breaks sorting, so normalise everything
        to UTC on load.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
