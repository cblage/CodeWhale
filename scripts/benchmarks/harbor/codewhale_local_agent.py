"""Harbor adapter that runs a local CodeWhale Linux binary artifact.

The stock CodeWhale Harbor adapter installs from npm, but npm may lag the local
release branch. This adapter uploads explicit Linux binaries into each
Terminal-Bench task container so benchmark rows identify the intended local
build.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path, PurePosixPath

from harbor.agents.installed.base import BaseInstalledAgent, CliFlag, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

CODEWHALE_LINUX_BIN_ENV = "CODEWHALE_LINUX_BIN"
CODEWHALE_TUI_LINUX_BIN_ENV = "CODEWHALE_TUI_LINUX_BIN"


class CodeWhaleLocalAgent(BaseInstalledAgent):
    """Run CodeWhale from host-built Linux binaries inside a Harbor task."""

    _OUTPUT_FILENAME = "codewhale.txt"
    _REMOTE_BIN = "/usr/local/bin/codewhale"
    _REMOTE_TUI_BIN = "/usr/local/bin/codewhale-tui"

    CLI_FLAGS = [
        CliFlag("max_subagents", cli="--max-subagents", type="int", default=None),
    ]

    def __init__(
        self,
        *args,
        local_binary_path: str | None = None,
        local_tui_binary_path: str | None = None,
        provider: str | None = None,
        reasoning_effort: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._local_binary_path = self._resolve_local_path(
            local_binary_path,
            CODEWHALE_LINUX_BIN_ENV,
        )
        self._local_tui_binary_path = self._resolve_local_path(
            local_tui_binary_path,
            CODEWHALE_TUI_LINUX_BIN_ENV,
        )
        self._provider_override = provider
        self._reasoning_effort = self._normalize_reasoning_effort(reasoning_effort)

    @staticmethod
    def _resolve_local_path(explicit: str | None, env_key: str) -> Path | None:
        value = explicit or os.environ.get(env_key)
        if value and value.strip():
            return Path(value.strip()).expanduser()
        return None

    @staticmethod
    def name() -> str:
        return "codewhale-local"

    def get_version_command(self) -> str | None:
        return f"{self._REMOTE_BIN} --version"

    def parse_version(self, stdout: str) -> str:
        text = stdout.strip()
        for line in text.splitlines():
            line = line.strip()
            if line:
                for prefix in ("codewhale-tui ", "codewhale-cli ", "codewhale "):
                    if line.lower().startswith(prefix):
                        return line[len(prefix) :]
                return line
        return text

    async def install(self, environment: BaseEnvironment) -> None:
        if self._local_binary_path is None:
            raise FileNotFoundError(
                "CodeWhale Linux binary path is required; pass "
                "local_binary_path=... or set CODEWHALE_LINUX_BIN."
            )
        if self._local_tui_binary_path is None:
            raise FileNotFoundError(
                "CodeWhale TUI Linux binary path is required; pass "
                "local_tui_binary_path=... or set CODEWHALE_TUI_LINUX_BIN."
            )
        if not self._local_binary_path.is_file():
            raise FileNotFoundError(f"CodeWhale Linux binary not found: {self._local_binary_path}")
        if not self._local_tui_binary_path.is_file():
            raise FileNotFoundError(
                f"CodeWhale TUI Linux binary not found: {self._local_tui_binary_path}"
            )

        await self.exec_as_root(
            environment,
            command=(
                "if command -v apt-get >/dev/null 2>&1; then "
                "apt-get update && "
                "ssl_pkg=''; "
                "if apt-cache show libssl3 >/dev/null 2>&1; then ssl_pkg=libssl3; "
                "elif apt-cache show libssl1.1 >/dev/null 2>&1; then ssl_pkg=libssl1.1; fi; "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y "
                "--no-install-recommends bash ca-certificates git ripgrep libdbus-1-3 $ssl_pkg; "
                "elif command -v apk >/dev/null 2>&1; then "
                "apk add --no-cache bash ca-certificates git ripgrep openssl dbus-libs; "
                "fi"
            ),
        )
        await environment.upload_file(self._local_binary_path, self._REMOTE_BIN)
        await environment.upload_file(self._local_tui_binary_path, self._REMOTE_TUI_BIN)
        await self.exec_as_root(
            environment,
            command=(
                f"chmod 755 {self._REMOTE_BIN} {self._REMOTE_TUI_BIN} && "
                f"ln -sf {self._REMOTE_BIN} /usr/local/bin/codew && "
                f"{self._REMOTE_BIN} --version && {self._REMOTE_TUI_BIN} --version"
            ),
        )

    def _provider_and_model(self) -> tuple[str, str]:
        raw = self.model_name or "deepseek/deepseek-v4-flash"
        if "/" in raw:
            provider, model = raw.split("/", 1)
        else:
            provider, model = "deepseek", raw
        if self._provider_override:
            provider = self._provider_override
        if provider == "openai-compatible":
            provider = "openai"
        return provider, model

    @staticmethod
    def _normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
        if reasoning_effort is None:
            return None
        normalized = reasoning_effort.strip().lower()
        aliases = {
            "none": "off",
            "disabled": "off",
            "false": "off",
            "medium": "high",
            "mid": "high",
            "maximum": "max",
            "xhigh": "max",
            "ultracode": "max",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"off", "high", "max"}:
            raise ValueError(
                "reasoning_effort must be one of off, high, or max "
                f"(got {reasoning_effort!r})"
            )
        return normalized

    @staticmethod
    def _key_env_for_provider(provider: str) -> str:
        return {
            "deepseek": "DEEPSEEK_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "zai": "ZAI_API_KEY",
            "z-ai": "ZAI_API_KEY",
        }.get(provider, f"{provider.replace('-', '_').upper()}_API_KEY")

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        provider, model = self._provider_and_model()
        key_env = self._key_env_for_provider(provider)
        api_key = self._get_env(key_env)
        if not api_key:
            raise ValueError(f"{key_env} is required for CodeWhale {provider} runs")

        pwd = await self.exec_as_agent(environment, "pwd")
        workspace = (pwd.stdout or "/workspace").strip() or "/workspace"
        output_path = PurePosixPath(EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME)
        cli_flags = self.build_cli_flags()
        extra_flags = f"{cli_flags} " if cli_flags else ""
        config_path = PurePosixPath("/tmp/codewhale-home/config.toml")
        config_arg = (
            f"--config {shlex.quote(config_path.as_posix())} "
            if self._reasoning_effort
            else ""
        )

        env: dict[str, str] = {
            key_env: api_key,
            "AWS_LC_SYS_NO_ASM": "1",
            "CODEWHALE_HOME": "/tmp/codewhale-home",
            "CODEWHALE_PROVIDER": provider,
            "CODEWHALE_MODEL": model,
        }
        for name in ("DEEPSEEK_BASE_URL", "CODEWHALE_BASE_URL", "OPENROUTER_BASE_URL"):
            value = self._get_env(name)
            if value:
                env[name] = value

        escaped_instruction = shlex.quote(instruction)
        config_lines = [
            f'provider = "{provider}"',
            f'default_text_model = "{model}"',
        ]
        if self._reasoning_effort:
            config_lines.append(f'reasoning_effort = "{self._reasoning_effort}"')
        write_config = "printf '%s\\n' " + " ".join(
            shlex.quote(line) for line in config_lines
        ) + f" > {shlex.quote(config_path.as_posix())}"
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {shlex.quote(EnvironmentPaths.agent_dir.as_posix())} "
                '"/tmp/codewhale-home" && '
                f"{write_config}"
            ),
            env=env,
            cwd=workspace,
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set +e; "
                f"{self._REMOTE_BIN} "
                f"{config_arg}"
                f"--provider {shlex.quote(provider)} "
                f"--model {shlex.quote(model)} "
                f"--workspace {shlex.quote(workspace)} "
                "--yolo "
                "exec --auto --output-format stream-json "
                f"{extra_flags}"
                f"-- {escaped_instruction} "
                f"2>&1 </dev/null | tee {shlex.quote(output_path.as_posix())}; "
                "status=${PIPESTATUS[0]}; "
                "rm -rf .codewhale .deepseek abs /tmp/codewhale-home; "
                "exit $status"
            ),
            env=env,
            cwd=workspace,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        output_path = self.logs_dir / self._OUTPUT_FILENAME
        if output_path.exists():
            context.metadata = {
                "codewhale_log": str(output_path),
                "reasoning_effort": self._reasoning_effort,
            }
