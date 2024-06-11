#!/usr/bin/env python
from argparse import ArgumentParser

from pydantic_settings import CliSettingsSource

from dolphin.workflows.config import DisplacementWorkflow


def main():
    parser = ArgumentParser()
    opts = parser
    opts.add_argument(
        "--print-empty",
        action="store_true",
        help="Flag to print a YAML file with only default filled to `outfile`.",
    )
    opts.add_argument(
        "-o",
        "--outfile",
        default="dolphin_config.yaml",
        help="Name of YAML configuration file to save to. Use '-' to write to stdout.",
    )
    parsed_args, extra_args = parser.parse_known_args()
    if parsed_args.print_empty:
        DisplacementWorkflow.print_yaml_schema(parsed_args.outfile)
        return

    # Set existing `parser` as `root_parser` object for the user defined settings source
    cli_settings = CliSettingsSource(
        DisplacementWorkflow,
        root_parser=parser,
        cli_use_nargs=True,
    )

    d = DisplacementWorkflow(_cli_settings_source=cli_settings(args=True))
    d.to_yaml(parsed_args.outfile)


if __name__ == "__main__":
    main()
