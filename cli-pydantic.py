#!/usr/bin/env python
# https://docs.pydantic.dev/latest/concepts/pydantic_settings/#command-line-support
# https://docs.pydantic.dev/latest/concepts/pydantic_settings/#creating-cli-applications
import sys
from pathlib import Path
from typing import Union

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, CliApp, SettingsError
from rich import print

from dolphin.workflows import DisplacementWorkflow


class Cli(
    BaseSettings,
    DisplacementWorkflow,
    cli_parse_args=True,
    cli_exit_on_error=False,
    cli_use_class_docs_for_groups=True,
    cli_implicit_flags=True,
):
    print_empty: bool = Field(
        False, description="Print an empty YAML files with default."
    )
    outfile: Union[Path, str] = Field(
        default=Path("dolphin_config.yaml"),
        help="Name of YAML configuration file to save to. Use '-' to write to stdout.",
    )


def main():
    try:
        idx = sys.argv.index("--outfile")
        sys.argv.pop(idx)
        outfile = sys.argv.pop(idx)
        output = sys.stdout if outfile == "-" else Path(outfile)
    except ValueError:
        output = Path("dolphin_config.yaml")

    if "--print-empty" in sys.argv or "--print_empty" in sys.argv:
        sys.argv.pop(sys.argv.index("--print-empty"))
        DisplacementWorkflow.print_yaml_schema(output)
        return

    cfg = CliApp.run(Cli, cli_exit_on_error=False)
    print(f"Saving configuration to {output!s}", file=sys.stderr)
    cfg.to_yaml(output)


if __name__ == "__main__":
    main()
