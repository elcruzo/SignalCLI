#!/usr/bin/env python3
"""
SignalCLI - LLM-Powered Knowledge CLI
Main CLI entry point
"""

import click
import asyncio
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.api.client import APIClient

console = Console()
logger = get_logger(__name__)

@click.command()
@click.argument('query', required=True)
@click.option('--schema', help='JSON schema file for structured output')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', default='json', type=click.Choice(['json', 'text', 'yaml']))
@click.option('--max-tokens', default=2048, help='Maximum tokens for response')
@click.option('--temperature', default=0.7, help='LLM temperature (0.0-1.0)')
@click.option('--api-url', default='http://localhost:8000', help='API server URL')
@click.version_option(version='1.0.0')
def main(query: str, schema: Optional[str], verbose: bool, output: Optional[str], 
         format: str, max_tokens: int, temperature: float, api_url: str):
    """
    SignalCLI - Query local LLM with RAG and structured JSON output
    
    Examples:
        signalcli "What is quantum computing?"
        signalcli "List programming languages" --schema schemas/list.json
        signalcli "Explain AI" --verbose --output result.json
    """
    # Set up logging
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Load configuration
    config = load_config()
    
    try:
        # Run the async query
        result = asyncio.run(process_query(
            query=query,
            schema_file=schema,
            api_url=api_url,
            max_tokens=max_tokens,
            temperature=temperature,
            output_format=format
        ))
        
        # Handle output
        if output:
            with open(output, 'w') as f:
                f.write(result)
            console.print(f"✅ Results saved to {output}")
        else:
            console.print(result)
            
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

async def process_query(query: str, schema_file: Optional[str], api_url: str,
                       max_tokens: int, temperature: float, output_format: str) -> str:
    """Process the query through the API"""
    
    # Load schema if provided
    schema = None
    if schema_file:
        import json
        with open(schema_file, 'r') as f:
            schema = json.load(f)
    
    # Initialize API client
    client = APIClient(base_url=api_url)
    
    # Show progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing query...", total=None)
        
        try:
            # Make the query
            response = await client.query(
                query=query,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            progress.update(task, description="✅ Query completed!")
            
            # Format output
            if output_format == 'json':
                import json
                return json.dumps(response, indent=2)
            elif output_format == 'yaml':
                import yaml
                return yaml.dump(response, default_flow_style=False)
            else:  # text
                return str(response.get('result', {}).get('answer', response))
                
        except Exception as e:
            progress.update(task, description="❌ Query failed!")
            raise e

if __name__ == '__main__':
    main()