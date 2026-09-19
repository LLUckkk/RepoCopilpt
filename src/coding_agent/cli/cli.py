import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer

from coding_agent.agent import (
    AgentLoop,
    AgentRunResult,
    AgentStepLimitError,
)
from coding_agent.cli.console_events import ConsoleEventHandler
from coding_agent.providers import (
    ModelProviderError,
    OpenAICompatibleProvider,
)
from coding_agent.tools import (
    ListDirectoryTool,
    ReadFileTool,
    SearchTextTool,
    ToolContext,
    ToolRegistry,
)

app = typer.Typer(
    name="coding-agent",
    help="A local coding agent operating inside a restricted workspace.",
    no_args_is_help=True,
    add_completion=False,
)


async def _execute_agent(
    *,
    task: str,
    workspace: Path,
    model: str,
    api_key: str,
    base_url: str | None,
    max_steps: int,
    verbose: bool,
) -> AgentRunResult:
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
        ]
    )

    agent = AgentLoop(
        provider=provider,
        registry=registry,
        context=ToolContext(workspace),
        max_steps=max_steps,
        event_handler=ConsoleEventHandler(verbose=verbose),
    )

    try:
        return await agent.run(task)
    finally:
        await provider.close()


@app.command()
def run(
    task: Annotated[
        str,
        typer.Argument(help="The coding task that the agent should investigate"),
    ],  # Argument是位置参数，默认必须
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
    typer.echo(f"Workspace: {workspace}")
    typer.echo(f"Model: {model_name}")
    typer.echo("Agent is working...")

    try:
        result = asyncio.run(
            _execute_agent(
                task=task,
                workspace=workspace,
                model=model_name,
                api_key=api_key,
                base_url=effective_base_url,
                max_steps=max_steps,
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

    typer.echo()
    typer.echo(result.final_text)
    typer.echo()
    typer.secho(
        f"Completed in {result.steps} model step(s).",
        fg=typer.colors.BRIGHT_BLACK,
    )
