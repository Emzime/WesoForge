use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use tokio::sync::mpsc;

#[derive(Debug)]
pub struct ShutdownController {
    forced: AtomicU8,
}

#[derive(Debug, Clone, Copy)]
pub enum ShutdownEvent {
    Graceful,
    Immediate,
}

#[derive(Debug, Clone, Copy)]
pub enum ClientEvent {
    Shutdown(ShutdownEvent),
    /// Toggle a GPU device by its ordinal (1..=9). `0` toggles all.
    ToggleGpuIndex(u8),
}

impl ShutdownController {
    pub fn new() -> Self {
        Self {
            forced: AtomicU8::new(0),
        }
    }

    pub fn bump_forced(&self) -> u8 {
        self.forced.fetch_add(1, Ordering::SeqCst) + 1
    }
}

pub fn spawn_ctrl_c_handler(
    shutdown: Arc<ShutdownController>,
    event_tx: mpsc::UnboundedSender<ClientEvent>,
) {
    tokio::spawn(async move {
        loop {
            if tokio::signal::ctrl_c().await.is_err() {
                return;
            }
            let n = shutdown.bump_forced();
            if n == 1 {
                let _ = event_tx.send(ClientEvent::Shutdown(ShutdownEvent::Graceful));
            } else {
                let _ = event_tx.send(ClientEvent::Shutdown(ShutdownEvent::Immediate));
                return;
            }
        }
    });
}
