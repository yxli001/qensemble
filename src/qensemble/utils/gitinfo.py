import subprocess


def get_git_sha(default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return default
