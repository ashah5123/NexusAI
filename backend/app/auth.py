from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from .models import (
    AuthSessionRead,
    InvitationCreate,
    InvitationRead,
    RegisterRequest,
    UserRead,
    WorkspaceCreate,
    WorkspaceRead,
)

DEFAULT_WORKSPACE_ID = "local"
SESSION_DAYS = 30


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    workspace_id: str
    role: str


class AuthError(ValueError):
    pass


class AuthRateLimitError(AuthError):
    pass


class AuthService:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _password(password: str, salt: bytes | None = None) -> tuple[str, str]:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
        return base64.b64encode(salt).decode(), base64.b64encode(digest).decode()

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _workspace(row: sqlite3.Row) -> WorkspaceRead:
        return WorkspaceRead(id=row["id"], name=row["name"], role=row["role"])

    def register(self, payload: RegisterRequest) -> tuple[str, AuthSessionRead]:
        email = payload.email.strip().lower()
        if "@" not in email:
            raise AuthError("Enter a valid email address")
        user_id = str(uuid4())
        salt, digest = self._password(payload.password)
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            try:
                connection.execute(
                    "INSERT INTO users (id, name, email, password_salt, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, payload.name, email, salt, digest, now),
                )
            except sqlite3.IntegrityError as exc:
                raise AuthError("An account with this email already exists") from exc
            member_count = connection.execute(
                "SELECT COUNT(*) FROM workspace_members WHERE workspace_id = ?",
                (DEFAULT_WORKSPACE_ID,),
            ).fetchone()[0]
            if member_count == 0:
                workspace_id = DEFAULT_WORKSPACE_ID
                role = "owner"
            else:
                workspace_id = str(uuid4())
                role = "owner"
                connection.execute(
                    "INSERT INTO workspaces (id, name, created_by, created_at) VALUES (?, ?, ?, ?)",
                    (workspace_id, f"{payload.name}'s workspace", user_id, now),
                )
            connection.execute(
                "INSERT INTO workspace_members (workspace_id, user_id, role, created_at) VALUES (?, ?, ?, ?)",
                (workspace_id, user_id, role, now),
            )
            connection.commit()
        token = self._create_session(user_id, workspace_id)
        return token, self.session(token)

    def login(self, email: str, password: str) -> tuple[str, AuthSessionRead]:
        normalized_email = email.strip().lower()
        with closing(self.connect()) as connection:
            now = datetime.now(UTC)
            attempt = connection.execute(
                "SELECT * FROM login_attempts WHERE email = ?", (normalized_email,)
            ).fetchone()
            if attempt and attempt["locked_until"] and attempt["locked_until"] > now.isoformat():
                raise AuthRateLimitError("Too many login attempts; try again later")
            row = connection.execute(
                "SELECT * FROM users WHERE email = ?", (normalized_email,)
            ).fetchone()
            valid = False
            if row is not None:
                salt = base64.b64decode(row["password_salt"])
                _, candidate = self._password(password, salt)
                valid = hmac.compare_digest(candidate, row["password_hash"])
            else:
                self._password(password, b"\0" * 16)
            if not valid:
                self._record_login_failure(connection, normalized_email, now, attempt)
                connection.commit()
                raise AuthError("Invalid email or password")
            connection.execute("DELETE FROM login_attempts WHERE email = ?", (normalized_email,))
            connection.commit()
            membership = connection.execute(
                """SELECT workspace_id FROM workspace_members WHERE user_id = ?
                ORDER BY created_at LIMIT 1""",
                (row["id"],),
            ).fetchone()
        if membership is None:
            raise AuthError("This account has no workspace")
        token = self._create_session(row["id"], membership["workspace_id"])
        return token, self.session(token)

    @staticmethod
    def _record_login_failure(
        connection: sqlite3.Connection,
        email: str,
        now: datetime,
        attempt: sqlite3.Row | None,
    ) -> None:
        window = timedelta(minutes=15)
        if attempt is None or now - datetime.fromisoformat(attempt["window_started_at"]) > window:
            failed_count = 1
            window_started = now
        else:
            failed_count = attempt["failed_count"] + 1
            window_started = datetime.fromisoformat(attempt["window_started_at"])
        locked_until = (now + window).isoformat() if failed_count >= 5 else None
        connection.execute(
            """INSERT INTO login_attempts
            (email, failed_count, window_started_at, locked_until, last_attempt_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET failed_count = excluded.failed_count,
                window_started_at = excluded.window_started_at,
                locked_until = excluded.locked_until, last_attempt_at = excluded.last_attempt_at""",
            (email, failed_count, window_started.isoformat(), locked_until, now.isoformat()),
        )

    def _create_session(self, user_id: str, workspace_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = datetime.now(UTC)
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO sessions
                (token_hash, user_id, workspace_id, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (self._token_hash(token), user_id, workspace_id,
                 (now + timedelta(days=SESSION_DAYS)).isoformat(), now.isoformat()),
            )
            connection.commit()
        return token

    def context(self, token: str | None) -> AuthContext | None:
        if not token:
            return None
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            row = connection.execute(
                """SELECT s.user_id, s.workspace_id, wm.role
                FROM sessions s JOIN workspace_members wm
                  ON wm.workspace_id = s.workspace_id AND wm.user_id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ?""",
                (self._token_hash(token), now),
            ).fetchone()
        return AuthContext(row["user_id"], row["workspace_id"], row["role"]) if row else None

    def session(self, token: str) -> AuthSessionRead:
        context = self.context(token)
        if context is None:
            raise AuthError("Session expired")
        with closing(self.connect()) as connection:
            user = connection.execute(
                "SELECT id, name, email FROM users WHERE id = ?", (context.user_id,)
            ).fetchone()
            rows = connection.execute(
                """SELECT w.id, w.name, wm.role FROM workspaces w
                JOIN workspace_members wm ON wm.workspace_id = w.id
                WHERE wm.user_id = ? ORDER BY w.name COLLATE NOCASE""",
                (context.user_id,),
            ).fetchall()
        workspaces = [self._workspace(row) for row in rows]
        current = next(item for item in workspaces if item.id == context.workspace_id)
        return AuthSessionRead(
            user=UserRead(**dict(user)), workspace=current, workspaces=workspaces
        )

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with closing(self.connect()) as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (self._token_hash(token),))
            connection.commit()

    def create_workspace(self, context: AuthContext, payload: WorkspaceCreate) -> WorkspaceRead:
        workspace_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                "INSERT INTO workspaces (id, name, created_by, created_at) VALUES (?, ?, ?, ?)",
                (workspace_id, payload.name, context.user_id, now),
            )
            connection.execute(
                "INSERT INTO workspace_members (workspace_id, user_id, role, created_at) VALUES (?, ?, 'owner', ?)",
                (workspace_id, context.user_id, now),
            )
            connection.commit()
        return WorkspaceRead(id=workspace_id, name=payload.name, role="owner")

    def switch_workspace(self, token: str, context: AuthContext, workspace_id: str) -> AuthSessionRead:
        with closing(self.connect()) as connection:
            membership = connection.execute(
                "SELECT 1 FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, context.user_id),
            ).fetchone()
            if membership is None:
                raise AuthError("Workspace access denied")
            connection.execute(
                "UPDATE sessions SET workspace_id = ? WHERE token_hash = ?",
                (workspace_id, self._token_hash(token)),
            )
            connection.commit()
        return self.session(token)

    def invite(self, context: AuthContext, payload: InvitationCreate) -> InvitationRead:
        if context.role != "owner":
            raise AuthError("Only workspace owners can invite members")
        token = secrets.token_urlsafe(24)
        expires = datetime.now(UTC) + timedelta(days=7)
        with closing(self.connect()) as connection:
            workspace = connection.execute(
                "SELECT name FROM workspaces WHERE id = ?", (context.workspace_id,)
            ).fetchone()
            connection.execute(
                """INSERT INTO workspace_invitations
                (token_hash, workspace_id, email, role, expires_at, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self._token_hash(token), context.workspace_id, payload.email.strip().lower(),
                 payload.role, expires.isoformat(), context.user_id, datetime.now(UTC).isoformat()),
            )
            connection.commit()
        return InvitationRead(
            token=token, email=payload.email.strip().lower(), workspace_name=workspace["name"],
            role=payload.role, expires_at=expires,
        )

    def accept_invitation(
        self, context: AuthContext, invitation_token: str, session_token: str
    ) -> AuthSessionRead:
        now = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            user = connection.execute("SELECT email FROM users WHERE id = ?", (context.user_id,)).fetchone()
            invite = connection.execute(
                """SELECT * FROM workspace_invitations
                WHERE token_hash = ? AND accepted_at IS NULL AND expires_at > ?""",
                (self._token_hash(invitation_token), now),
            ).fetchone()
            if invite is None or invite["email"] != user["email"]:
                raise AuthError("Invitation is invalid, expired, or belongs to another email")
            connection.execute(
                """INSERT INTO workspace_members (workspace_id, user_id, role, created_at)
                VALUES (?, ?, ?, ?) ON CONFLICT(workspace_id, user_id) DO UPDATE SET role = excluded.role""",
                (invite["workspace_id"], context.user_id, invite["role"], now),
            )
            connection.execute(
                "UPDATE workspace_invitations SET accepted_at = ? WHERE token_hash = ?",
                (now, invite["token_hash"]),
            )
            connection.commit()
        return self.switch_workspace(
            session_token, context, invite["workspace_id"]
        )
