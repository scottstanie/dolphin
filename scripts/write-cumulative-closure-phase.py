#!/usr/bin/env python
"""Sum closure phase rasters and convert to equivalent displacement.

This script processes the closure_phase_*.tif rasters contained in the
`interferograms/` subfolder (generated when `write_closure_phase=True`)
and outputs cumulative closure phase converted to meters.

Output units are meters, with positive values indicating apparent motion
towards the sensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from opera_utils import get_dates

from dolphin import constants, io

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
    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    # Determine wavelength
    if args.wavelength is not None:
        wavelength = args.wavelength
    else:
        wavelength = SENSOR_WAVELENGTHS[args.sensor]

    # Find input files
    input_files = sorted(args.directory.glob(args.glob_pattern))
    if not input_files:
        msg = f"No files matching '{args.glob_pattern}' found in {args.directory}"
        raise FileNotFoundError(msg)

    # Determine output directory
    output_dir = args.output_dir if args.output_dir is not None else args.directory

    reader = io.RasterStackReader.from_file_list(input_files)

    running_sum = np.zeros(reader.shape[1:], dtype="float64")
    for idx, fin in enumerate(reader.file_list):
        running_sum += reader[idx, :, :].filled(0).squeeze().astype("float64")
        date_str = get_dates(fin)[1].strftime("%Y%m%d")
        fname = f"cumulative_closure_phase_{date_str}.tif"
        fout = output_dir / fname

        # Convert phase to displacement: displacement = -wavelength * phase / (4 * pi)
        # Negative sign flips convention so positive = motion towards sensor
        displacement = running_sum.astype("float32") * wavelength / -4 / np.pi

        io.write_arr(
            arr=displacement,
            output_name=fout,
            like_filename=fin,
            options=io.EXTRA_COMPRESSED_TIFF_OPTIONS,
        )
        print(f"Wrote {fout}")


if __name__ == "__main__":
    main()
