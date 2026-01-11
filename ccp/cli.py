"""
CCP CLI - Context Collapse Parser Command Line Interface

Main entry point for all CCP commands.
"""

import click

from ccp import __version__
from ccp.commands.validate import validate_data
from ccp.commands.train import train
from ccp.commands.eval import eval_cmd
from ccp.commands.eval_semantic import eval_semantic_cmd
from ccp.commands.inference import inference
from ccp.commands.history import history


@click.group()
@click.version_option(version=__version__, prog_name="ccp")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.pass_context
def cli(ctx, verbose):
    """CCP - Context Collapse Parser CLI

    A command-line tool for fine-tuning and running the Phoenix
    Context Collapse Parser for GTM intent recognition.

    \b
    Commands:
      validate-data  Validate training data schema
      train          Fine-tune the CCP model
      eval           Evaluate model performance (exact field matching)
      eval-semantic  Evaluate GTM understanding (semantic scoring)
      inference      Run inference on a single prompt
      history        View evaluation history

    \b
    Examples:
      ccp validate-data --file data/ccp_training_with_reasoning.jsonl
      ccp train --epochs 3 --resume-from latest
      ccp eval --quick
      ccp eval-semantic --quick --compare-with-exact
      ccp inference "Show me my best accounts"
      ccp history --compare 2
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# Register commands
cli.add_command(validate_data)
cli.add_command(train)
cli.add_command(eval_cmd, name="eval")
cli.add_command(eval_semantic_cmd, name="eval-semantic")
cli.add_command(inference)
cli.add_command(history)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
