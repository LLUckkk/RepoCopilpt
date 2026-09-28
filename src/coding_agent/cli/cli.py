import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from coding_agent.agent import (
    AgentLoop,
    AgentRunResult,
    AgentStepLimitError,
)
from coding_agent.cli.console_approval import ConsoleApprovalHandler
from coding_agent.cli.console_events import ConsoleEventHandler
from coding_agent.cli.ui import (
    render_final_answer,
    render_header,
    render_interactive_help,
    render_session_notice,
)
from coding_agent.providers import (
    ModelProviderError,
    OpenAICompatibleProvider,
)
from coding_agent.tools import (
    CreateDirectoryTool,
    CreateFileTool,
    FindFilesTool,
    GitDiffTool,
    GitStatusTool,
    ListDirectoryTool,
    ReadFileTool,
    ReplaceTextTool,
    RunCommandTool,
    SearchTextTool,
    ToolContext,
    ToolRegistry,
)

app = typer.Typer(
    name="coding-agent",
    help="A local coding agent operating inside a restricted workspace.",
    no_args_is_help=False,
    add_completion=False,
)


INTERACTIVE_PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "ansicyan bold",
    }
)


async def _run_interactive_session(
    agent: AgentLoop,
) -> None:
    input_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
    )
    agent_session = agent.create_session()
    render_interactive_help()

    while True:
        try:
            user_input = await input_session.prompt_async(
                FormattedText(
                    [
                        ("class:prompt", "❯ "),
                    ]
                ),
                style=INTERACTIVE_PROMPT_STYLE,
            )
        except KeyboardInterrupt:
            render_session_notice(
                "Input cancelled. Use /exit to leave.",
                style="warning",
            )
            continue
        except EOFError:
            render_session_notice("Session ended.")
            return

        task = user_input.strip()
        if not task:
            continue

        command = task.casefold()
        if command in {"/exit", "/quit"}:
            render_session_notice("Session ended.")
            return
        if command == "/help":
            render_interactive_help()
            continue
        if command == "/clear":
            agent_session = agent.create_session()
            render_session_notice(
                "Conversation context cleared.",
                style="success",
            )
            continue
        if command.startswith("/"):
            render_session_notice(
                f"Unknown command: {task}",
                style="warning",
            )
            continue

        try:
            result = await agent.run(task, session=agent_session)
        except ModelProviderError as exc:
            render_session_notice(
                f"Model provider error: {exc}",
                style="error",
            )
            continue
        except AgentStepLimitError as exc:
            render_session_notice(
                str(exc),
                style="error",
            )
            continue

        render_final_answer(
            text=result.final_text,
            steps=result.steps,
        )


async def _execute_agent(
    *,
    task: str | None,
    workspace: Path,
    model: str,
    api_key: str,
    base_url: str | None,
    max_steps: int,
    max_context_tokens: int | None,
    verbose: bool,
) -> AgentRunResult | None:
    provider = OpenAICompatibleProvider(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    registry = ToolRegistry(
        [
            ListDirectoryTool(),
            ReadFileTool(),
            SearchTextTool(),
            ReplaceTextTool(),
            RunCommandTool(),
            CreateFileTool(),
            CreateDirectoryTool(),
            FindFilesTool(),
            GitStatusTool(),
            GitDiffTool(),
        ]
    )

    event_handler = ConsoleEventHandler(verbose=verbose)

    agent = AgentLoop(
        provider=provider,
        registry=registry,
        context=ToolContext(
            workspace_root=workspace, approval_handler=ConsoleApprovalHandler()
        ),
        max_steps=max_steps,
        max_context_tokens=max_context_tokens,
        event_handler=event_handler,
    )

    try:
        # 兼容两种模式，如果第一次输入了指令，就按照单次运行，否则就是启动session
        if task is not None:
            return await agent.run(task)

        await _run_interactive_session(agent)
        return None
    finally:
        event_handler.close()
        await provider.close()


@app.command()
def run(
    task: Annotated[
        str | None,
        typer.Argument(
            help="Optional coding task. If omitted, starts an interactive session."
        ),
    ] = None,  # Argument是位置参数，默认必须
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            "-w",
            help="Workspace in which the agent is allowed to operate.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),  # 表示选项，需要--
    ] = Path("."),
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=(
                "Model name. Falls back to the CODING_AGENT_MODEL environment variable."
            ),
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help=(
                "Optional OpenAI-compatible API base URL. Falls back to "
                "CODING_AGENT_BASE_URL or OPENAI_BASE_URL."
            ),
        ),
    ] = None,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            help="Maximum number of model turns.",
            min=1,
            max=200,
        ),
    ] = 20,
    max_context_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-context-tokens",
            help=(
                "Estimated input-token budget. "
                "Warns at 80 percent and compacts older "
                "completed interactions when necessary."
            ),
            min=1_000,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Display tool outputs in addition to tool calls.",
        ),
    ] = False,
) -> None:
    api_key = os.getenv("CODING_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = model or os.getenv("CODING_AGENT_MODEL")
    effective_base_url = (
        base_url or os.getenv("CODING_AGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    )

    if not api_key:
        typer.secho(
            "Missing api key. Set CODING_AGENT_API_KEY or OPENAI_API_KEY.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    if not model_name:
        typer.secho(
            "Missing model. Pass --model or set CODING_AGENT_MODEL.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    render_header(workspace=workspace, model=model_name)

    try:
        result = asyncio.run(
            _execute_agent(
                task=task,
                workspace=workspace,
                model=model_name,
                api_key=api_key,
                base_url=effective_base_url,
                max_steps=max_steps,
                max_context_tokens=max_context_tokens,
                verbose=verbose,
            )
        )
    except ModelProviderError as exc:
        typer.secho(
            f"Model provider error: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except AgentStepLimitError as exc:
        typer.secho(
            str(exc),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt as exc:
        typer.echo("\nCancelled.", err=True)
        raise typer.Exit(code=130) from exc

    if result is not None:
        render_final_answer(
            text=result.final_text,
            steps=result.steps,
        )
