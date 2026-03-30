from typing import Any

import wandb


def log_bundle_as_artifact(
    run: Any | None,
    bundle_dir: str,
    name: str,
    artifact_type: str,
    aliases: list[str] | None = None,
) -> None:
    if run is None:
        return

    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_dir(bundle_dir)
    run.log_artifact(artifact, aliases=aliases or [])
