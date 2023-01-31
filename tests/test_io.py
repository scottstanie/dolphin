from pathlib import Path

import pytest

from dolphin import io


def test_format_nc_filename():
    expected = 'NETCDF:"/usr/19990101/20200303_20210101.nc":"//variable"'
    assert (
        io.format_nc_filename("/usr/19990101/20200303_20210101.nc", "variable")
        == expected
    )

    # check on Path
    assert (
        io.format_nc_filename(Path("/usr/19990101/20200303_20210101.nc"), "variable")
        == expected
    )

    # check non-netcdf file
    assert (
        io.format_nc_filename("/usr/19990101/20200303_20210101.tif")
        == "/usr/19990101/20200303_20210101.tif"
    )
    assert (
        io.format_nc_filename("/usr/19990101/20200303_20210101.int", "ignored")
        == "/usr/19990101/20200303_20210101.int"
    )

    with pytest.raises(ValueError):
        # Missing the subdataset name
        io.format_nc_filename("/usr/19990101/20200303_20210101.nc")
