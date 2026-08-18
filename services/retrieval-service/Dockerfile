FROM rust:slim AS builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y pkg-config libssl-dev g++ build-essential
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

FROM debian:trixie-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libssl-dev ca-certificates libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/target/release/retrieval-service /app/retrieval-service

EXPOSE 8002
CMD ["/app/retrieval-service"]
