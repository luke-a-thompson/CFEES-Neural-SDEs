import json
import tomllib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import seali


class Experiments(StrEnum):
    SPD = "spd"


class Devices(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class Solvers(StrEnum):
    GL2 = "gl2"
    CFEES25 = "cfees25"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: Experiments
    epochs: int
    batch_size: int
    learning_rate: float
    loss_beta: float
    seed: int
    device: Devices
    # model hyperparams
    hidden_dim: int
    ctx_dim: int
    n_steps: int
    dt: float
    solver: Solvers
    diffusion_scale: float
    min_eigenvalue: float
    # output
    skip_plots: bool


def make_config(
    *,
    experiment: Experiments = Experiments.SPD,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    loss_beta: float = 0.0,
    seed: int = 0,
    device: Devices = Devices.GPU,
    hidden_dim: int = 128,
    ctx_dim: int = 64,
    n_steps: int = 5,
    dt: float = 0.2,
    solver: Solvers = Solvers.CFEES25,
    diffusion_scale: float = 1.0,
    min_eigenvalue: float = 1e-6,
    skip_plots: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=experiment,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_beta=loss_beta,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        ctx_dim=ctx_dim,
        n_steps=n_steps,
        dt=dt,
        solver=solver,
        diffusion_scale=diffusion_scale,
        min_eigenvalue=min_eigenvalue,
        skip_plots=skip_plots,
    )


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return ExperimentConfig(
        experiment=Experiments(data["experiment"]),
        device=Devices(data["device"]),
        solver=Solvers(data["solver"]),
        epochs=data["epochs"],
        batch_size=data["batch_size"],
        learning_rate=data["learning_rate"],
        loss_beta=data["loss_beta"],
        seed=data["seed"],
        hidden_dim=data["hidden_dim"],
        ctx_dim=data["ctx_dim"],
        n_steps=data["n_steps"],
        dt=data["dt"],
        diffusion_scale=data["diffusion_scale"],
        min_eigenvalue=data["min_eigenvalue"],
        skip_plots=data["skip_plots"],
    )


def _serialize_config(config: ExperimentConfig) -> str:
    return json.dumps(asdict(config), indent=2, sort_keys=True)


HELP = seali.Help(
    help="""
    Build an experiment config and print it as JSON.

    $USAGE

    $OPTIONS_AND_FLAGS
    """,
    style=seali.Style(heading=seali.BOLD),
    arguments={
        "experiment": "Experiment preset to encode into the config.",
        "epochs": "Number of training epochs.",
        "batch_size": "Mini-batch size.",
        "learning_rate": "Optimizer learning rate.",
        "loss_beta": "Weight on the auxiliary loss term.",
        "seed": "Random seed.",
        "device": "Preferred runtime device.",
        "hidden_dim": "Hidden dimension of the neural SDE.",
        "ctx_dim": "Context dimension.",
        "n_steps": "Number of SDE integration steps.",
        "dt": "SDE time step size.",
        "solver": "SDE solver.",
        "diffusion_scale": "Scale of the diffusion coefficient.",
        "min_eigenvalue": "Minimum eigenvalue clamp for SPD projections.",
        "skip_plots": "Skip saving diagnostic plots.",
        "output": "Optional path to write the JSON config to.",
    },
    option_prompts={
        "experiment": "experiment",
        "epochs": "int",
        "batch_size": "int",
        "learning_rate": "float",
        "loss_beta": "float",
        "seed": "int",
        "device": "device",
        "hidden_dim": "int",
        "ctx_dim": "int",
        "n_steps": "int",
        "dt": "float",
        "solver": "solver",
        "diffusion_scale": "float",
        "min_eigenvalue": "float",
        "skip_plots": "flag",
        "output": "path",
    },
)


@seali.command(help=HELP)
def main(
    *,
    experiment: Experiments = Experiments.SPD,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    loss_beta: float = 0.0,
    seed: int = 0,
    device: Devices = Devices.GPU,
    hidden_dim: int = 128,
    ctx_dim: int = 64,
    n_steps: int = 5,
    dt: float = 0.2,
    solver: Solvers = Solvers.CFEES25,
    diffusion_scale: float = 1.0,
    min_eigenvalue: float = 1e-6,
    skip_plots: bool = False,
    output: Path | None = None,
):
    """Build an experiment config and print it as JSON."""
    config = make_config(
        experiment=experiment,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_beta=loss_beta,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        ctx_dim=ctx_dim,
        n_steps=n_steps,
        dt=dt,
        solver=solver,
        diffusion_scale=diffusion_scale,
        min_eigenvalue=min_eigenvalue,
        skip_plots=skip_plots,
    )
    payload = _serialize_config(config)

    if output is None:
        print(payload)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
