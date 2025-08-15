"""CLI interface for adapter registry operations."""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .registry import AdapterRegistry

console = Console()


@click.group()
def cli():
    """Memory adapter registry CLI."""
    pass


@cli.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table',
              help='Output format')
@click.option('--capability', '-c', help='Filter by capability')
def list(format: str, capability: Optional[str]):
    """List registered adapters and their capabilities."""
    # Import adapters to trigger registration
    try:
        from . import memory_store, faiss_store, file_store
    except ImportError as e:
        click.echo(f"Warning: Could not import all adapters: {e}", err=True)
    
    all_adapters = AdapterRegistry.list_all()
    
    if not all_adapters:
        console.print("[yellow]No adapters registered[/yellow]")
        return
    
    # Filter by capability if specified
    if capability:
        filtered_adapters = {}
        for name, caps in all_adapters.items():
            if capability in caps:
                filtered_adapters[name] = caps
        all_adapters = filtered_adapters
        
        if not all_adapters:
            console.print(f"[yellow]No adapters found with capability '{capability}'[/yellow]")
            return
    
    if format == 'json':
        import json
        # Convert sets to lists for JSON serialization
        json_data = {name: list(caps) for name, caps in all_adapters.items()}
        click.echo(json.dumps(json_data, indent=2))
    else:
        # Rich table format
        table = Table(title="Registered Memory Adapters")
        table.add_column("Adapter Name", style="cyan")
        table.add_column("Capabilities", style="green")
        
        for name, caps in sorted(all_adapters.items()):
            caps_str = ", ".join(sorted(caps))
            table.add_row(name, caps_str)
        
        console.print(table)


@cli.command()
@click.argument('capability')
def find(capability: str):
    """Find adapters that provide a specific capability."""
    # Import adapters to trigger registration
    try:
        from . import memory_store, faiss_store, file_store
    except ImportError as e:
        click.echo(f"Warning: Could not import all adapters: {e}", err=True)
    
    adapters = AdapterRegistry.by_capability(capability)
    
    if not adapters:
        console.print(f"[yellow]No adapters found with capability '{capability}'[/yellow]")
        console.print("\nAvailable capabilities:")
        all_caps = set()
        for caps in AdapterRegistry.list_all().values():
            all_caps.update(caps)
        for cap in sorted(all_caps):
            console.print(f"  • {cap}")
    else:
        console.print(f"[green]Adapters with capability '{capability}':[/green]")
        for adapter in adapters:
            console.print(f"  • {adapter}")


@cli.command()
def capabilities():
    """List all available capability types."""
    console.print("[bold]Standard Capability Types:[/bold]")
    
    capabilities_info = {
        AdapterRegistry.CAPABILITY_VECTOR: "Vector-based similarity search",
        AdapterRegistry.CAPABILITY_SEMANTIC: "Semantic understanding of content", 
        AdapterRegistry.CAPABILITY_PERSISTENT: "Persistent storage to disk",
        AdapterRegistry.CAPABILITY_CHEAP: "Low cost operations",
        AdapterRegistry.CAPABILITY_FAST: "Fast response times",
        AdapterRegistry.CAPABILITY_SCALABLE: "Handles large datasets",
        AdapterRegistry.CAPABILITY_SEARCHABLE: "Full-text search capabilities",
        AdapterRegistry.CAPABILITY_TEMPORAL: "Time-based querying"
    }
    
    for capability, description in capabilities_info.items():
        console.print(f"  [cyan]{capability}[/cyan] - {description}")


if __name__ == "__main__":
    cli()