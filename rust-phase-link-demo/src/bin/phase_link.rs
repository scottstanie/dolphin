//! End-to-end CLI: read a stack of complex SLC GeoTIFFs, run EVD phase
//! linking with a sliding window, write one float32 phase GeoTIFF per date.
//!
//! Usage:
//!
//! ```bash
//! phase-link \
//!     --input-glob 'synthetic-demo/slcs/2*.tif' \
//!     --output-dir synthetic-demo/phase-linked \
//!     --half-row 5 --half-col 5
//! ```
//!
//! Built only with `--features gdal-io` (requires libgdal).

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use gdal::raster::{Buffer, GdalDataType};
use gdal::{Dataset, DriverManager};
use ndarray::{s, Array2, Array3};
use num_complex::Complex;

use phase_link_demo::{pipeline::run_in_memory, C32};

#[derive(Parser, Debug)]
#[command(about = "Minimal EVD phase linking demo")]
struct Args {
    /// Glob pattern for input SLC GeoTIFFs (sorted lexicographically).
    #[arg(long)]
    input_glob: String,

    /// Directory to write per-date float32 phase rasters into.
    #[arg(long)]
    output_dir: PathBuf,

    /// Half window in rows (full window = 2·half_row + 1).
    #[arg(long, default_value_t = 5)]
    half_row: usize,

    /// Half window in cols.
    #[arg(long, default_value_t = 5)]
    half_col: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut paths: Vec<PathBuf> = glob::glob(&args.input_glob)
        .with_context(|| format!("invalid glob: {}", args.input_glob))?
        .filter_map(std::result::Result::ok)
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(anyhow!("no files matched {}", args.input_glob));
    }
    println!("Found {} SLCs", paths.len());

    // Read first SLC to size the stack and capture geo metadata.
    let (first, geo) = read_complex_slc(&paths[0])?;
    let (rows, cols) = first.dim();
    let nslc = paths.len();
    println!("Stack shape: ({nslc}, {rows}, {cols})");

    let mut stack = Array3::<C32>::zeros((nslc, rows, cols));
    stack.slice_mut(s![0, .., ..]).assign(&first);
    for (i, p) in paths.iter().enumerate().skip(1) {
        let (slc, _) = read_complex_slc(p)?;
        if slc.dim() != (rows, cols) {
            return Err(anyhow!(
                "shape mismatch at {}: {:?} != {:?}",
                p.display(),
                slc.dim(),
                (rows, cols)
            ));
        }
        stack.slice_mut(s![i, .., ..]).assign(&slc);
    }

    println!(
        "Running EVD phase linking with half-window ({}, {})…",
        args.half_row, args.half_col
    );
    let phase = run_in_memory(stack.view(), args.half_row, args.half_col);

    std::fs::create_dir_all(&args.output_dir)?;
    for (i, p) in paths.iter().enumerate() {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("slc");
        let out_path = args.output_dir.join(format!("{stem}.phase.tif"));
        let band = phase.slice(s![i, .., ..]);
        write_float_geotiff(&out_path, band.view(), &geo)?;
    }
    println!("Wrote {nslc} phase rasters to {}", args.output_dir.display());
    Ok(())
}

#[derive(Clone)]
struct GeoMeta {
    geo_transform: [f64; 6],
    projection: String,
}

fn read_complex_slc(path: &PathBuf) -> Result<(Array2<C32>, GeoMeta)> {
    let ds = Dataset::open(path).with_context(|| format!("opening {}", path.display()))?;
    let band = ds.rasterband(1)?;
    let (cols, rows) = band.size();

    let buf: Buffer<Complex<f32>> = match band.band_type() {
        GdalDataType::CFloat32 => band.read_band_as::<Complex<f32>>()?,
        other => {
            return Err(anyhow!(
                "{}: expected CFloat32 SLC, got {:?}",
                path.display(),
                other
            ));
        }
    };

    let data = buf.into_shape_and_vec().1;
    let arr = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| anyhow!("shape error reading {}: {e}", path.display()))?;

    let geo = GeoMeta {
        geo_transform: ds.geo_transform().unwrap_or([0., 1., 0., 0., 0., 1.]),
        projection: ds.projection(),
    };
    Ok((arr, geo))
}

fn write_float_geotiff(
    path: &PathBuf,
    data: ndarray::ArrayView2<f32>,
    geo: &GeoMeta,
) -> Result<()> {
    let driver = DriverManager::get_driver_by_name("GTiff")?;
    let (rows, cols) = data.dim();
    let mut ds = driver.create_with_band_type::<f32, _>(path, cols, rows, 1)?;
    let _ = ds.set_geo_transform(&geo.geo_transform);
    let _ = ds.set_projection(&geo.projection);

    let buf_vec: Vec<f32> = data.iter().copied().collect();
    let buffer = Buffer::new((cols, rows), buf_vec);
    let mut band = ds.rasterband(1)?;
    band.write::<f32>((0, 0), (cols, rows), &buffer)?;
    Ok(())
}
