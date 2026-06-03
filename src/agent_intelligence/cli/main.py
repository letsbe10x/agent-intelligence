"""`ai-cli` — drive agents from the shell."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.table import Table

from agent_intelligence.core.context import AgentContext
from agent_intelligence.observability.receipts import Receipt, ReceiptStore
from agent_intelligence.react.runner import run_react_agent
from agent_intelligence.registry.registry import registry
from agent_intelligence.tools.registry import tool_registry

console = Console()


@click.group(help="agent-intelligence — agentic framework where the LLM owns decisions.")
def cli() -> None: ...


@cli.command("list-tools")
def list_tools() -> None:
    """List all registered tools."""
    import agent_intelligence.tools.builtins  # noqa: F401
    names = tool_registry.list_names()
    if not names:
        console.print("[yellow]No tools registered.[/yellow]")
        return
    t = Table(title="Registered tools")
    t.add_column("Name", style="cyan")
    t.add_column("Description")
    for n in names:
        tool = tool_registry.get(n)
        t.add_row(n, tool.description[:80] + ("…" if len(tool.description) > 80 else ""))
    console.print(t)


@cli.command("list-providers")
def list_providers() -> None:
    """List registered LLM providers."""
    t = Table(title="Registered providers")
    t.add_column("Name", style="cyan")
    t.add_column("Module")
    for n in registry.list_providers():
        cls = registry.get_provider(n)
        t.add_row(n, f"{cls.__module__}.{cls.__name__}")
    console.print(t)


@cli.command("describe-tool")
@click.argument("name")
def describe_tool(name: str) -> None:
    """Show a tool's input + output schemas."""
    import agent_intelligence.tools.builtins  # noqa: F401
    tool = tool_registry.get(name)
    console.print(Panel.fit(f"[bold cyan]{name}[/bold cyan]\n{tool.description}", title="Tool"))
    console.print("\n[bold]Input schema[/bold]")
    console.print(RichJSON.from_data(tool.InputModel.model_json_schema()))
    console.print("\n[bold]Output schema[/bold]")
    console.print(RichJSON.from_data(tool.OutputModel.model_json_schema()))


@cli.command("react-run")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--org-id", default=None)
@click.option("--bet-id", default=None)
@click.option(
    "--receipts-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Persist receipts to this directory.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write final output JSON to this file.",
)
@click.option("--steps/--no-steps", default=True, help="Print step-by-step trace.")
def react_run(
    config_path: str,
    input_path: str,
    org_id: str | None,
    bet_id: str | None,
    receipts_dir: str | None,
    output: str | None,
    steps: bool,
) -> None:
    """Run a YAML ReAct agent end-to-end. The LLM picks tools step by step."""
    import agent_intelligence.tools.builtins  # noqa: F401 — auto-register
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    store = ReceiptStore(path=receipts_dir) if receipts_dir else ReceiptStore(path=None)
    context = AgentContext(
        org_id=org_id,
        bet_id=bet_id,
        metadata={
            k: v for k, v in payload.items()
            if k.startswith("known_") or k in ("org_id", "bet_id")
        },
    )

    async def _go() -> None:
        result = await run_react_agent(
            config_path, payload, context=context, receipt_store=store
        )
        if steps:
            for s in result.steps:
                console.print("---")
                console.print(json.dumps(s, indent=2, default=str))
        out = {
            "output": result.output,
            "receipt_id": result.receipt_id,
            "receipt_hash": result.receipt_hash,
            "iterations": result.iterations,
            "halted_reason": result.halted_reason,
            "cost_usd": result.cost_usd,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "wallclock_s": result.wallclock_s,
        }
        rendered = json.dumps(out, indent=2, default=str)
        if output:
            Path(output).write_text(rendered, encoding="utf-8")
            console.print(f"[green]wrote {output}[/green]")
        else:
            console.print(rendered)

    asyncio.run(_go())


@cli.command("verify-receipt")
@click.argument("receipt_path", type=click.Path(exists=True, dir_okay=False))
def verify_receipt(receipt_path: str) -> None:
    data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    receipt = Receipt.from_dict(data)
    if receipt.verify():
        console.print(f"[green]✓ {receipt.receipt_id} verified[/green]  hash: {receipt.payload_hash}")
        sys.exit(0)
    else:
        console.print(f"[red]✗ {receipt.receipt_id} TAMPERED[/red]")
        sys.exit(2)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
