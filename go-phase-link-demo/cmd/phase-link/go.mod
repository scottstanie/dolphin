// Separate module so the heavy GDAL/cgo dependency tree only affects this
// CLI, not the core algorithm packages. Mirrors the Rust `gdal-io` feature.
module github.com/scottstanie/dolphin/go-phase-link-demo/cmd/phase-link

go 1.24

require (
	github.com/airbusgeo/godal v0.0.17
	github.com/scottstanie/dolphin/go-phase-link-demo v0.0.0
)

replace github.com/scottstanie/dolphin/go-phase-link-demo => ../..
