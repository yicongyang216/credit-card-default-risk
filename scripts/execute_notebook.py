"""Execute saved notebooks with this Python, without global kernel registration."""

import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = root / ".cache" / "jupyter"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(runtime))
    os.environ.setdefault("IPYTHONDIR", str(root / ".cache" / "ipython"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    for path in sorted((root / 'notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        # Use this environment rather than a globally registered Python.
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=120, startup_timeout=30,
                                resources={"metadata": {"path": str(root / "notebooks")}})
        try:
            client.execute()
        finally:
            if client.kc is not None:
                client.kc.stop_channels()
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(notebook, path)
        print(f"Executed {path.name} successfully", flush=True)


if __name__ == "__main__":
    main()
