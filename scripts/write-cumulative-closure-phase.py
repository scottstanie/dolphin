#!/usr/bin/env python
"""Sum closure phase rasters and convert to equivalent displacement.

Thin command-line wrapper around `dolphin.closure.write_cumulative_closure_phase`,
which the displacement workflow now runs automatically after stitching when
`phase_linking.write_closure_phase` is set. Use this to (re)build the rasters for
an existing run. Output units are meters, positive toward the sensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dolphin import constants

SENSOR_WAVELENGTHS: dict[str, float] = {
    "sentinel-1": constants.SENTINEL_1_WAVELENGTH,
    "nisar-l": constants.NISAR_L_WAVELENGTH,
    "nisar-s": constants.NISAR_S_WAVELENGTH,
    "uavsar": constants.UAVSAR_WAVELENGTH,
    "capella": constants.CAPELLA_WAVELENGTH,
}


def get_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing closure_phase_*.tif files (e.g., interferograms/).",
    )

    wavelength_group = parser.add_mutually_exclusive_group()
    wavelength_group.add_argument(
        "-w",
        "--wavelength",
        type=float,
        help="Radar wavelength in meters for phase-to-displacement conversion.",
    )
    wavelength_group.add_argument(
        "-s",
        "--sensor",
        choices=list(SENSOR_WAVELENGTHS.keys()),
        default="sentinel-1",
        help=(
            "Sensor name to use predefined wavelength. "
            "Choices: %(choices)s. Default: %(default)s."
        ),
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for cumulative files. "
            "Default: same directory as input files."
        ),
    )
    parser.add_argument(
        "-g",
        "--glob-pattern",
        default="closure_phase_*.tif",
        help="Glob pattern to match closure phase files. Default: %(default)s.",
    )

    return parser


def main(args: argparse.Namespace | None = None) -> None:
    """Run the cumulative closure phase computation."""
    from dolphin.closure import write_cumulative_closure_phase

    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    wavelength = (
        args.wavelength
        if args.wavelength is not None
        else SENSOR_WAVELENGTHS[args.sensor]
    )
    input_files = sorted(args.directory.glob(args.glob_pattern))
    if not input_files:
        msg = f"No files matching '{args.glob_pattern}' found in {args.directory}"
        raise FileNotFoundError(msg)
    output_dir = args.output_dir if args.output_dir is not None else args.directory
    written = write_cumulative_closure_phase(
        input_files, output_dir=output_dir, wavelength=wavelength, overwrite=True
    )
    for f in written:
        print(f"Wrote {f}")


if __name__ == "__main__":
    main()
