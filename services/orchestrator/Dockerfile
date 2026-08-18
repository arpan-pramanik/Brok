FROM rust:slim AS builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y pkg-config libssl-dev
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

FROM debian:trixie-slim
WORKDIR /app
RUN apt-get update && apt-get install -y openssl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/target/release/orchestrator /app/orchestrator

EXPOSE 8000
CMD ["/app/orchestrator"]
