from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


DB_PATH = Path(os.getenv("NEXUSAI_DB_PATH", "./data/nexusai.db"))
UPLOAD_DIR = Path(os.getenv("NEXUSAI_UPLOAD_DIR", "./data/uploads"))


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_backup(output: Path) -> Path:
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nexusai-backup-") as temporary:
        staging = Path(temporary)
        database_copy = staging / "database.sqlite3"
        with sqlite3.connect(DB_PATH) as source, sqlite3.connect(database_copy) as target:
            source.backup(target)
        files = {"database.sqlite3": checksum(database_copy)}
        if UPLOAD_DIR.is_dir():
            for path in sorted(UPLOAD_DIR.rglob("*")):
                if path.is_file():
                    files[f"uploads/{path.relative_to(UPLOAD_DIR).as_posix()}"] = checksum(path)
        manifest = {
            "format": "nexusai-backup-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        with tarfile.open(output, "w:gz") as archive:
            archive.add(staging / "manifest.json", arcname="manifest.json")
            archive.add(database_copy, arcname="database.sqlite3")
            if UPLOAD_DIR.is_dir():
                archive.add(UPLOAD_DIR, arcname="uploads", recursive=True)
    return output


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute() or ".." in path.parts or member.issym() or member.islnk()
            or not (member.isfile() or member.isdir())
        ):
            raise ValueError(f"Unsafe archive member: {member.name}")
    return members


def _extract(archive: tarfile.TarFile, destination: Path) -> None:
    members = _safe_members(archive)
    try:
        archive.extractall(destination, members=members, filter="data")
    except TypeError:  # Python 3.11 before extraction filters were available.
        archive.extractall(destination, members=members)


def verify_backup(archive_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="nexusai-verify-") as temporary:
        staging = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            _extract(archive, staging)
        manifest_path = staging / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Backup manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "nexusai-backup-v1":
            raise ValueError("Unsupported backup format")
        for relative, expected in manifest.get("files", {}).items():
            path = staging / relative
            if not path.is_file() or checksum(path) != expected:
                raise ValueError(f"Backup checksum failed: {relative}")
        with sqlite3.connect(staging / "database.sqlite3") as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise ValueError(f"Database integrity check failed: {result}")
        return manifest


def restore_backup(archive_path: Path, force: bool) -> None:
    if not force:
        raise ValueError("Restore requires --force; stop the API before restoring")
    verify_backup(archive_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    with tempfile.TemporaryDirectory(prefix="nexusai-restore-") as temporary:
        staging = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            _extract(archive, staging)
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        if DB_PATH.exists():
            shutil.copy2(DB_PATH, DB_PATH.with_suffix(f".before-restore-{timestamp}.sqlite3"))
        shutil.copy2(staging / "database.sqlite3", DB_PATH)
        restored_uploads = staging / "uploads"
        UPLOAD_DIR.parent.mkdir(parents=True, exist_ok=True)
        if UPLOAD_DIR.exists():
            shutil.move(UPLOAD_DIR, UPLOAD_DIR.with_name(f"uploads.before-restore-{timestamp}"))
        if restored_uploads.is_dir():
            shutil.copytree(restored_uploads, UPLOAD_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="NexusAI backup and recovery utility")
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("archive", type=Path)
    restore = commands.add_parser("restore")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "backup":
        print(create_backup(args.output))
    elif args.command == "verify":
        print(json.dumps(verify_backup(args.archive), indent=2))
    else:
        restore_backup(args.archive, args.force)
        print("Restore complete")


if __name__ == "__main__":
    main()
