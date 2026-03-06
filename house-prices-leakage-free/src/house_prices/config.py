from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]
    raw_dir: Path = field(init=False)
    artifact_dir: Path = field(init=False)

    random_state: int = 42
    outer_folds: int = 5
    inner_folds: int = 5

    def __post_init__(self) -> None:
        self.raw_dir = self.project_root / "data" / "raw"
        self.artifact_dir = self.project_root / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
