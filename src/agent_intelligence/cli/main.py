"""``ai-cli`` — list, describe, and invoke agents from the shell.

Subcommands:
    list-agents              List registered agents.
    list-providers           List registered providers.
    describe <agent>         Show input/output/params schemas for an agent.
    run <config> <input>     Load config, build agent, run with JSON input.
    verify-receipt <path>    Recompute hash of a persisted receipt; report drift.
"""

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
from agent_intelligence.registry.registry import registry
from agent_intelligence.runner import build_agent

console = Console()


@click.group(help="agent-intelligence — configurable, multi-model agentic framework.")
def cli() -> None: ...


@cli.command("list-agents")
def list_agents() -> None:
    """List registered agents."""
    names = registry.list_agents()
    if not names:
        console.print("[yellow]No agents registered.[/yellow]")
        return
    t = Table(title="Registered agents")
    t.add_column("Name", style="cyan")
    t.add_column("Module")
    for n in names:
        cls = registry.get_agent(n)
        t.add_row(n, f"{cls.__module__}.{cls.__name__}")
    console.print(t)


@cli.command("list-providers")
def list_providers() -> None:
    """List registered LLM providers."""
    names = registry.list_providers()
    t = Table(title="Registered providers")
    t.add_column("Name", style="cyan")
    t.add_column("Module")
    for n in names:
        cls = registry.get_provider(n)
        t.add_row(n, f"{cls.__module__}.{cls.__name__}")
    console.print(t)


@cli.command("describe")
@click.argument("agent_name")
def describe(agent_name: str) -> None:
    """Show schemas for an agent."""
    cls = registry.get_agent(agent_name)
    console.print(
        Panel.fit(
            f"[bold cyan]{agent_name}[/bold cyan]\n"
            f"{cls.__module__}.{cls.__name__}",
            title="Agent",
        )
    )
    for label, model in (
        ("Input", cls.InputModel),
        ("Output", cls.OutputModel),
        ("Params", cls.ParamsModel),
    ):
        console.print(f"\n[bold]{label} schema[/bold]")
        console.print(RichJSON.from_data(model.model_json_schema()))


@cli.command("run")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--org-id", default=None)
@click.option("--bet-id", default=None)
@click.option(
    "--receipts-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Persist receipts to this directory. Overrides config.observability.receipts_path.",
)
@click.option("--output", type=click.Path(dir_okay=False), default=None, help="Write output JSON to this file.")
def run(
    config_path: str,
    input_path: str,
    org_id: str | None,
    bet_id: str | None,
    receipts_dir: str | None,
    output: str | None,
) -> None:
    """Load config, build agent, run with JSON input from a file."""
    input_payload = json.loads(Path(input_path).read_text(encoding="utf-8"))

    receipt_store = ReceiptStore(path=receipts_dir) if receipts_dir else None
    agent = build_agent(config_path, receipt_store=receipt_store)
    context = AgentContext(org_id=org_id, bet_id=bet_id)

    async def _go() -> None:
        result = await agent.run(input_payload, context)
        out_data = {
            "output": result.output.model_dump(),
            "receipt_id": result.receipt.receipt_id,
            "receipt_hash": result.receipt.payload_hash,
            "wallclock_s": result.wallclock_s,
            "cost_usd": result.cost_usd,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "model_calls": result.model_calls,
        }
        rendered = json.dumps(out_data, indent=2, default=str)
        if output:
            Path(output).write_text(rendered, encoding="utf-8")
            console.print(f"[green]wrote {output}[/green]")
        else:
            console.print(rendered)

    asyncio.run(_go())


@cli.command("verify-receipt")
@click.argument("receipt_path", type=click.Path(exists=True, dir_okay=False))
def verify_receipt(receipt_path: str) -> None:
    """Recompute the hash of a receipt JSON and report whether it matches."""
    data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    receipt = Receipt.from_dict(data)
    if receipt.verify():
        console.print(f"[green]✓ Receipt {receipt.receipt_id} verified.[/green]")
        console.print(f"  hash: {receipt.payload_hash}")
        sys.exit(0)
    else:
        console.print(f"[red]✗ Receipt {receipt.receipt_id} failed verification.[/red]")
        sys.exit(2)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
