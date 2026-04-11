SPEED_OF_LIGHT = 299_792_458  # meters / second

SENTINEL_1_FREQUENCY = 5.405e9  # Hz
SENTINEL_1_WAVELENGTH = SPEED_OF_LIGHT / SENTINEL_1_FREQUENCY  # meters

NISAR_L_FREQUENCY = 1_257_476_312  # Hz, matches UAVSAR L-band (~0.23840 m)
NISAR_S_FREQUENCY = 3.2e9  # Hz
NISAR_L_WAVELENGTH = SPEED_OF_LIGHT / NISAR_L_FREQUENCY  # meters
NISAR_S_WAVELENGTH = SPEED_OF_LIGHT / NISAR_S_FREQUENCY  # meters

# NISAR L-band center frequencies by acquisition mode, per NISAR D-102269
# Figure 3-1. Keyed by the 4-digit MODE code embedded in the granule filename
# (slot 8 of the underscore-split name per §3.4). Each value is
# (freqA_center_Hz, freqB_center_Hz); `None` means the mode has no freqB.
# Full 77 MHz mode is intentionally omitted: its freqA center equals
# NISAR_L_FREQUENCY above, so the generic constant fallback already covers it.
NISAR_L_MODE_CENTERS_HZ: dict[str, tuple[int, int | None]] = {
    "4005": (1_229_000_000, 1_293_500_000),  # 40+5 MHz split
    "2005": (1_229_000_000, 1_293_500_000),  # 20+5 MHz split
    "2020": (1_229_000_000, 1_286_000_000),  # 20+20 MHz split
    "0505": (1_221_500_000, 1_236_500_000),  # 5+5 MHz split
}

UAVSAR_WAVELENGTH = 0.238403545  # meters
UAVSAR_FREQUENCY = SPEED_OF_LIGHT / UAVSAR_WAVELENGTH

CAPELLA_FREQUENCY = 9.65e9  # Hz (approximate X-band)
CAPELLA_WAVELENGTH = SPEED_OF_LIGHT / CAPELLA_FREQUENCY  # meters
