#![forbid(unsafe_code)]
#![deny(unreachable_pub)]
#![deny(missing_docs)]

//! In-process engine for `bbr-client` (job leasing, proof computation, submission).

/// Public API for the engine crate.
pub mod api;

mod backend;
mod cpu_affinity;
mod cuda_backend;
mod engine;
mod inflight;
mod worker;

// Re-export public API types for consumers (client, GUI, etc.).
pub use api::{
    start_engine,
    EngineConfig,
    EngineEvent,
    EngineHandle,
    JobOutcome,
    JobSummary,
    StatusSnapshot,
    WorkerSnapshot,
    WorkerStage,
    GpuBackend,
};
