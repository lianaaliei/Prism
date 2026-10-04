# LUMEN

A zero-dependency, multithreaded path tracer written in Rust.

- Diffuse, metal (with fuzz), glass (Schlick + total internal reflection), emissive materials
- Depth of field (thin-lens camera)
- Iterative tracing with Russian roulette
- ACES filmic tone-mapping + gamma correction
- Hand-written PNG encoder (CRC32 + Adler-32, no crates)
- Row-parallel rendering with std::thread

## Run

    cargo run --release -- [width] [samples_per_pixel] [output.png]
    cargo run --release -- 1920 256 hd.png
