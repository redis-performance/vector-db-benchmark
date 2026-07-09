# Weaviate gRPC protos (vendored)

`v1/*.proto` are copied verbatim from weaviate/weaviate `grpc/proto/v1` (tag v1.29.1)
and used to generate `src/bin/vector_db_benchmark/engine/weaviate_grpc.rs`.

The generated Rust is checked in (no `protoc`/`build.rs` at build time). To regenerate
after updating the protos, run tonic-build against these files with include root `proto/`:

    tonic_build::configure()
        .build_server(false)
        .compile_protos(&["proto/v1/weaviate.proto"], &["proto"])?;

then wrap the emitted `weaviate.v1.rs` in `pub mod weaviate_v1 { ... }` and replace
`engine/weaviate_grpc.rs`.
