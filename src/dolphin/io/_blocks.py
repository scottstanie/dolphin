"""Module for handling blocked/decimated input and output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from dolphin._types import HalfWindow, Strides
from dolphin.utils import compute_out_shape

# 1. iterate without overlap over output, decimated array
#   - Start with an offset, shrink the size, both by `half_window//strides`
#   - since we want every one wot be full, just skipping the incomplete blocks
# 2. map the current output pixels to their input locations (using strides)
# 3. pad the input block (using half window)

__all__ = [
    "BlockIndices",
    "StridedBlockManager",
    "iter_blocks",
]


@dataclass(frozen=True)
class BlockIndices:
    """Class holding slices for 2D array access."""

    row_start: int
    row_stop: Optional[int]  # Can be None if we want slice(0, None)
    col_start: int
    col_stop: Optional[int]

    @classmethod
    def from_slices(cls, row_slice: slice, col_slice: slice) -> BlockIndices:
        return cls(
            row_start=row_slice.start,
            row_stop=row_slice.stop,
            col_start=col_slice.start,
            col_stop=col_slice.stop,
        )

    @property
    def row_slice(self) -> slice:
        return slice(self.row_start, self.row_stop)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_start, self.col_stop)

    # Allows user to unpack like (row_slice, col_slice) = block_index
    def __iter__(self):
        return iter((self.row_slice, self.col_slice))


def iter_blocks(
    arr_shape: tuple[int, int],
    block_shape: tuple[int, int],
    overlaps: tuple[int, int] = (0, 0),
    start_offsets: tuple[int, int] = (0, 0),
    end_margin: tuple[int, int] = (0, 0),
) -> Iterator[BlockIndices]:
    """Create a generator to get indexes for accessing blocks of a raster.

    Parameters
    ----------
    arr_shape : tuple[int, int]
        (num_rows, num_cols), full size of array to access
    block_shape : tuple[int, int]
        (height, width), size of blocks to load
    overlaps : tuple[int, int], default = (0, 0)
        (row_overlap, col_overlap), number of pixels to re-include from
        the previous block after sliding
    start_offsets : tuple[int, int], default = (0, 0)
        Offsets from top left to start reading from
    end_margin : tuple[int, int], default = (0, 0)
        Margin to avoid at the bottom/right of array

    Yields
    ------
    BlockIndices
        Iterator of BlockIndices, which can be unpacked into
        (slice(row_start, row_stop), slice(col_start, col_stop))

    Examples
    --------
        >>> list(iter_blocks((180, 250), (100, 100)))
        [BlockIndices(row_start=0, row_stop=100, col_start=0, col_stop=100), \
BlockIndices(row_start=0, row_stop=100, col_start=100, col_stop=200), \
BlockIndices(row_start=0, row_stop=100, col_start=200, col_stop=250), \
BlockIndices(row_start=100, row_stop=180, col_start=0, col_stop=100), \
BlockIndices(row_start=100, row_stop=180, col_start=100, col_stop=200), \
BlockIndices(row_start=100, row_stop=180, col_start=200, col_stop=250)]
        >>> list(map(tuple, iter_blocks((180, 250), (100, 100), overlaps=(10, 10))))
        [(slice(0, 100, None), slice(0, 100, None)), (slice(0, 100, None), \
slice(90, 190, None)), (slice(0, 100, None), slice(180, 250, None)), \
(slice(90, 180, None), slice(0, 100, None)), (slice(90, 180, None), \
slice(90, 190, None)), (slice(90, 180, None), slice(180, 250, None))]
    """
    total_rows, total_cols = arr_shape
    height, width = block_shape
    row_overlap, col_overlap = overlaps
    start_row_offset, start_col_offset = start_offsets
    last_row = total_rows - end_margin[0]
    last_col = total_cols - end_margin[1]

    if height is None:
        height = total_rows
    if width is None:
        width = total_cols

    # Check we're not moving backwards with the overlap:
    if row_overlap >= height and height != total_rows:
        msg = f"{row_overlap = } must be less than block height {height}"
        raise ValueError(msg)
    if col_overlap >= width and width != total_cols:
        msg = f"{col_overlap = } must be less than block width {width}"
        raise ValueError(msg)

    # Set up the iterating indices
    cur_row = start_row_offset
    cur_col = start_col_offset
    while cur_row < last_row:
        while cur_col < last_col:
            row_stop = min(cur_row + height, last_row)
            col_stop = min(cur_col + width, last_col)
            # yield (slice(cur_row, row_stop), slice(cur_col, col_stop))
            yield BlockIndices(cur_row, row_stop, cur_col, col_stop)

            cur_col += width
            if cur_col < last_col:  # dont bring back if already at edge
                cur_col -= col_overlap

        cur_row += height
        if cur_row < last_row:
            cur_row -= row_overlap
        cur_col = start_col_offset  # reset back to the starting offset


def unstride_center(decimated_index: int, stride: int) -> int:
    """Compute the inverse of a striding operation, finding the full-res area center.

    Note that for even strides, there are two valid centers;
    we return the larger index. i.e. for `stride=2`, then
        unstride_center(0, 2) = 1

    Parameters
    ----------
    decimated_index : int
        Index from output, decimated array
    stride : int
        Striding/decimation factor

    Returns
    -------
    int
        Center of corresponding pixel in full-res array.
    """
    return decimated_index * stride + stride // 2


def unstride_center_slice(decimated_slice: slice, stride: int) -> slice:
    """Get the slice pointing at the centers of a full-res slice."""
    full_res_start = unstride_center(decimated_slice.start, stride)
    if decimated_slice.stop is not None:
        last_decimated_idx = decimated_slice.stop - 1
        full_res_end = unstride_center(last_decimated_idx, stride) + 1
    else:
        full_res_end = None
    return slice(full_res_start, full_res_end)


def unstride_center_block(
    decimated_block: BlockIndices,
    strides: Strides,
) -> BlockIndices:
    """Grow slices in `BlockIndices` to undo a striding operation.

    This is so we can go back from the smaller, strided/decimated grid
    indices to the original, full-res grid indices.

    `decimated_block` is the smaller one which was made by taking `strides`
    from the larger block.

    Here we translate the decimated indexes to the full-res region *center* indices.

    Uses `unstride_center_slice` to do the actual work on each dimension.

    Parameters
    ----------
    decimated_block : BlockIndices
        Slices for an smaller, strided array.
    strides : tuple[int, int] or Strides(y, x)
        Decimation factor in y and x which was used for `in_block`

    Returns
    -------
    BlockIndices
        Output slices for larger array
    """
    # Just treat each dim using the 1d function
    row_slice, col_slice = decimated_block
    full_row_slice = unstride_center_slice(row_slice, strides.y)
    full_col_slice = unstride_center_slice(col_slice, strides.x)
    return BlockIndices.from_slices(full_row_slice, full_col_slice)


"""

when going for the “upsampling” step, I want to grab the colored
block regions and ignore the half window again.

the “trimming” part cam from the fact that I calculate the part to load
using strides and half window.

BlockManager(arr_shape=(10, 10), block_shape=(15, 15),
             strides=Strides(y=3, x=3), half_window=HalfWindow(y=4, x=4))
[
    col_start=1, col_stop=2),
    col_start=1, col_stop=-1),
    col_start=0, col_stop=9),
    col_start=4, col_stop=5)
]
slc_stack.shape = (11, 9, 9)
Out[257]: (1, 1, 11, 11)

Now, the out was one OutSize pixel, the in relevant SLC pixels are (3, 4, 5)
(that's why the final slice is slice(4, 5). it's talking about the one full-res center)
BUT i don't wanna read in from the main slc usince `slice(3, 6)`, which would be easy:
out_slice=(1, 2), so slice(1*3, 2*3) = slice(3, 6) is what i want.
Instead, i have the real data I read which was (0, 9). how can i get what I want?

I need a trimming slice... the (0, 9) was made by
    1. get the output indices (thats slice(1, 2) here. i only want the out index [1])
    2. find the SLC center pixels those correspond to (slice(4, 5), or full-res index 4)
    3. Pad that slice with the half window ( +/-4 pixels, which makes it slice(0, 9))

So now to get to the *relative* slice to trim so results are the same as slice(3, 6):
    1. i should figure out slice(3,6), which is _unstrided_full_cover(output_slice, 3)
    2. figure the offset relative to the slice(0, 9)
#TODO: MAKE AN ERROR IF THEY PASS HALF WINDOW LESS THAN STRIDES//2
(verify- is this def a problem? or just weird? it's ignoring input data...)
(It WOULD break the assumption of _get_relative_offset_slice, because we think that the
full_padded_slice will be bigger than _unstrided_full_cover, but it wouldnt it
strides//2 > half_window)
"""


def get_output_size(in_size: int, stride: int, half_window: int) -> int:
    """Get the size of valid data for one dimension from the strided output.

    Uses the half_window to find the amount of padding to ignore at the edges.

    Parameters
    ----------
    in_size : int
        Size of input array
    stride : int
        Striding/decimation factor
    half_window : int
        Half window size

    Returns
    -------
    int
        Size of output array
    """
    # out_rows = (rows - 2 * half_row) // row_strides + 1
    # out_cols = (cols - 2 * half_col) // col_strides + 1
    return (in_size - half_window * 2) // stride + 1


def unstride_full_cover_slice(
    decimated_slice: slice,
    stride: int,
) -> slice:
    """Dilate a slice by a stride factor to cover the whole full-res region.

    Parameters
    ----------
    decimated_slice : slice
        Slice from output, decimated array
    stride : int
        Striding/decimation factor

    Returns
    -------
    slice
        slice which covers the whole full-res region corresponding
        to `decimated_slice`
    """
    return slice(stride * decimated_slice.start, stride * decimated_slice.stop)


def unstride_full_cover_block(
    decimated_block: BlockIndices, strides: Strides
) -> BlockIndices:
    """Grow slices in `BlockIndices` to undo a striding operation.

    Parameters
    ----------
    decimated_block : BlockIndices
        BlockIndices from output, decimated array
    strides : tuple[int, int] or Strides(y, x)
        Decimation factor in y and x which was used for `in_block`

    Returns
    -------
    BlockIndices
        BlockIndices which covers the whole full-res region corresponding
        to `decimated_block`
    """
    row_slice, col_slice = decimated_block
    row_s, col_s = strides
    return BlockIndices.from_slices(
        unstride_full_cover_slice(row_slice, row_s),
        unstride_full_cover_slice(col_slice, col_s),
    )


def _get_relative_offset_slice(
    output_slice: slice, full_padded_slice: slice, strides: int
) -> slice:
    """Get the offset of the output slice relative to the full-res slice."""
    full_res_inner = unstride_full_cover_slice(output_slice, strides)
    # We assume that half_window > strides // 2, which makes the start
    #  of `full_res_inner` larger than the start of `full_padded_slice`.:
    #   full_res_inner =    slice(3, 6)
    #   full_padded_slice = slice(0, 9)
    relative_start = full_res_inner.start - full_padded_slice.start
    relative_end = full_res_inner.stop - full_padded_slice.stop
    return slice(relative_start, relative_end)


def _get_relative_offset_block(
    output_block: BlockIndices,
    full_padded_block: BlockIndices,
    strides: Strides,
) -> BlockIndices:
    """Get the offset of the output block relative to the full-res block."""
    row_slice, col_slice = output_block
    row_offset_slice = _get_relative_offset_slice(
        row_slice, full_padded_block.row_slice, strides.y
    )
    col_offset_slice = _get_relative_offset_slice(
        col_slice, full_padded_block.col_slice, strides.x
    )
    return BlockIndices.from_slices(row_offset_slice, col_offset_slice)


def get_slice_length(s: slice, data_size: int = 1_000_000):
    """Get the size of a slice of data.

    Uses `slice.indices` to avoid making any dummy data.
    Assumes that 1. data is larger than slice, and 2. size is
    less than `data_size`.
    """
    return len(range(*s.indices(data_size)))


def pad_block(in_block: BlockIndices, margins: tuple[int, int]) -> BlockIndices:
    """Pad `in_block` by the (row_margin, col_margin) pixels in `margins`.

    Will clip the row/column slice starts to be >= 0.

    Parameters
    ----------
    in_block : BlockIndices
        Slices original array block.
    margins : dict[int, int]
        Number of pixels to extend `in_block` in the (row, col) directions.
        The margins are subtracted from the beginning and added to the end,
        so the block size grows by (2 * row_margin, 2 * col_margin)

    Returns
    -------
    BlockIndices
        Output slices for larger block

    Raises
    ------
    ValueError
        If the block is too small to be padded by the given margins
        (leads to start < 0)
    """
    r_margin, c_margin = margins
    r_slice, c_slice = in_block
    if r_slice.start - r_margin < 0:
        msg = f"{r_slice = }, but {r_margin = }"
        raise ValueError(msg)
    if c_slice.start - c_margin < 0:
        msg = f"{c_slice = }, but {c_margin = }"
        raise ValueError(msg)
    return BlockIndices(
        # max(r_slice.start - r_margin, 0),
        r_slice.start - r_margin,
        r_slice.stop + r_margin,
        # max(c_slice.start - c_margin, 0),
        c_slice.start - c_margin,
        c_slice.stop + c_margin,
    )


@dataclass
class StridedBlockManager:
    """Class to handle slicing/trimming overlapping blocks with strides."""

    arr_shape: tuple[int, int]
    """(row, col) of the full-res 2D image"""
    block_shape: tuple[int, int]
    """(row, col) size of each input block to operate on at one time"""
    strides: Strides = field(default_factory=lambda: Strides(1, 1))
    """Decimation/downsampling factor in y/row and x/column direction"""
    half_window: HalfWindow = field(default_factory=lambda: HalfWindow(0, 0))
    """For multi-looking iterations, size of the (full-res) half window
    in y/row and x/column direction.
    Used to find `overlaps` between blocks and `start_offset`/`end_margin` for
    `iter_blocks`."""

    def iter_blocks(
        self,
    ) -> Iterator[tuple[BlockIndices, BlockIndices, BlockIndices, BlockIndices]]:
        """Iterate over the input/output block indices.

        Yields
        ------
        output_block : BlockIndices
            The current slices for the output raster
        full_res_input : BlockIndices
            Slices used to load the full-res input data
        full_res_relative_teim : BlockIndices
            Slices to use on a full-res input block to remove nodata border pixels
            caused by the half_window padding.
            These slices are relative (e.g. slice(1, -1)).
        full_res_output : BlockIndices
            Slices which point to the position within the full-res data without padding.
        """
        # trimming_block = self.get_trimming_block()
        for out_block in self.iter_outputs():
            # First undo the stride/decimation factor
            input_no_padding = unstride_center_block(out_block, strides=self.strides)
            full_res_input = pad_block(
                input_no_padding, margins=(self.half_window.y, self.half_window.x)
            )
            # now get the relative to trim away the edges of `full_res_input`
            full_res_relative_trim = _get_relative_offset_block(
                out_block, full_res_input, strides=self.strides
            )
            # And form the full-res location where we'll write an output
            full_res_output = unstride_full_cover_block(out_block, strides=self.strides)

            yield (out_block, full_res_input, full_res_relative_trim, full_res_output)

    def _get_out_nodata_size(self, direction: str) -> int:
        return round(
            getattr(self.half_window, direction) / getattr(self.strides, direction)
        )

    @property
    def output_shape(self) -> tuple[int, int]:
        return compute_out_shape(self.arr_shape, self.strides)

    @property
    def out_block_shape(self) -> tuple[int, int]:
        return compute_out_shape(self.block_shape, self.strides)

    @property
    def input_padding_shape(self) -> tuple[int, int]:
        """Amount of extra padding the input blocks need.

        Depends on the window size, and the strides.
        """
        return (
            self.strides.y * self._get_out_nodata_size("y"),
            self.strides.x * self._get_out_nodata_size("x"),
        )

    @property
    def output_margin(self) -> tuple[int, int]:
        """The output margins that we ignore while iterating.

        Depends on the half window and strides.

        The `half_window` is in full-res (input) coordinates (which would be the
        amount to skip with no striding), so the output margin size is smaller
        """
        return (self._get_out_nodata_size("y"), self._get_out_nodata_size("x"))

    def iter_outputs(self) -> Iterator[BlockIndices]:
        yield from iter_blocks(
            arr_shape=self.output_shape,
            block_shape=self.out_block_shape,
            overlaps=(0, 0),  # We're not overlapping in the *output* grid
            start_offsets=self.output_margin,
            end_margin=self.output_margin,
        )

    # def get_trimming_block(self) -> BlockIndices:
    #     """Compute the slices which trim output nodata values.

    #     When the BlockIndex gets dilated (using `strides`) and padded (using
    #     `half_window`), the result will have nodata around the edges.
    #     The size of the nodata pixels in the full-res block is just
    #         (half_window['y'], half_window['x'])
    #     In the output (strided) coordinates, the number of nodata pixels is
    #     shrunk by how many strides are taken.

    #     Note that this is independent of which block we're on; the number of
    #     nodata pixels on the border is always the same.
    #     """
    #     row_nodata = self._get_out_nodata_size("y")
    #     col_nodata = self._get_out_nodata_size("x")
    #     # Extra check if we have no trimming to do: use slice(0, None)
    #     row_end = -row_nodata if row_nodata > 0 else None
    #     col_end = -col_nodata if col_nodata > 0 else None
    #     return BlockIndices(row_nodata, row_end, col_nodata, col_end)
