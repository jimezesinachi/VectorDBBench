"""Generate isolated synchronous bindings from an Ahnlich checkout.

Run with grpcio-tools==1.67.1 and protobuf==5.29.5, using an Ahnlich checkout.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ahnlich-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.ahnlich_root.resolve()
    adapter = Path(__file__).resolve().parents[1]
    proto_dir = source / "protos"
    output = adapter / "_grpc"
    for package, expected in {"grpcio-tools": "1.67.1", "protobuf": "5.29.5"}.items():
        if importlib.metadata.version(package) != expected:
            message = f"Generation requires {package}=={expected}"
            raise RuntimeError(message)

    files = set()

    def visit(name: str):
        if name in files:
            return
        files.add(name)
        content = (source / "protos" / name).read_text()
        for dependency in re.findall(r'^import "([^"]+)";', content, flags=re.MULTILINE):
            visit(dependency)

    visit("services/db_service.proto")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    hashes = {}
    for name in sorted(files):
        hashes[name] = hashlib.sha256((proto_dir / name).read_bytes()).hexdigest()

    subprocess.run(
        [sys.executable, "-m", "grpc_tools.protoc", f"-I{proto_dir}", f"--python_out={output}"]
        + [str(proto_dir / name) for name in sorted(files)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--grpc_python_out={output}",
            str(proto_dir / "services/db_service.proto"),
        ],
        check=True,
    )
    prefix = "vectordb_bench.backend.clients.ahnlich._grpc"
    for generated in output.rglob("*.py"):
        content = generated.read_text()
        content = re.sub(
            r"^from ([a-z_]+) import (\w+_pb2) as (\w+)$",
            lambda match: f"from {prefix}.{match[1]} import {match[2]} as {match[3]}",
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(
            r"^import (\w+_pb2) as (\w+)$",
            lambda match: f"from {prefix} import {match[1]} as {match[2]}",
            content,
            flags=re.MULTILINE,
        )
        generated.write_text(content)
    for directory in [output, *(p for p in output.rglob("*") if p.is_dir())]:
        (directory / "__init__.py").write_text('"""Generated Ahnlich protobuf bindings; do not edit manually."""\n')
    revision = subprocess.check_output([shutil.which("git"), "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "repository": "https://github.com/deven96/ahnlich",
        "revision": revision,
        "grpcio_tools": "1.67.1",
        "protobuf": "5.29.5",
        "proto_sha256": hashes,
    }
    (adapter / "proto_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(source / "LICENSE", adapter / "AHNLICH_LICENSE")


if __name__ == "__main__":
    main()
