//! Core error types used across planner, backend, and executor code.

use crate::pipeline::{BufferSlot, ElementKind, KernelKind};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Standard result alias used by `braids`.
pub type BraidResult<T> = Result<T, BraidError>;

#[derive(Clone, Debug)]
/// Error type for core stack, planner, backend, and packet operations.
pub enum BraidError {
    /// Job was cancelled cooperatively.
    Cancelled,
    /// A stack-local job id was not found.
    UnknownJob,
    /// Executor or backend runtime is shutting down.
    ExecutorShutdown,
    /// Backend did not know how to prepare the requested kernel kind.
    BackendRejectedKernel(KernelKind),
    /// Requested packet buffer slot was missing.
    MissingBuffer(BufferSlot),
    /// Packet buffer slot existed with the wrong element type.
    InvalidBufferType {
        /// Slot that had the wrong type.
        slot: BufferSlot,
        /// Expected element kind for that slot.
        expected: ElementKind,
    },
    /// Planner encountered a duplicate identifier.
    DuplicateId {
        /// Identifier kind label.
        kind: &'static str,
        /// Duplicate identifier value.
        id: String,
    },
    /// Planner encountered a missing referenced identifier.
    MissingReference {
        /// Identifier kind label.
        kind: &'static str,
        /// Owner identifier value.
        id: String,
        /// Missing referenced identifier.
        reference: String,
    },
    /// Planner encountered an empty required scope.
    EmptyScope {
        /// Identifier kind label.
        kind: &'static str,
        /// Identifier value with empty scope.
        id: String,
    },
    /// Generic invalid compiled-plan or pipeline-layout error.
    InvalidSpec(String),
    /// Generic message error.
    Message(String),
    /// Shared synchronization primitive was poisoned.
    Poisoned(&'static str),
}

impl BraidError {
    /// Helper for creating a poisoned shared-state error.
    pub fn poisoned(name: &'static str) -> Self {
        Self::Poisoned(name)
    }
}

impl Display for BraidError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(f, "job cancelled"),
            Self::UnknownJob => write!(f, "unknown job"),
            Self::ExecutorShutdown => write!(f, "executor is shut down"),
            Self::BackendRejectedKernel(kind) => {
                write!(f, "backend rejected kernel kind {}", kind)
            }
            Self::MissingBuffer(slot) => write!(f, "missing buffer at slot {}", slot),
            Self::InvalidBufferType { slot, expected } => {
                write!(
                    f,
                    "invalid buffer type at slot {} expected {:?}",
                    slot, expected
                )
            }
            Self::DuplicateId { kind, id } => write!(f, "duplicate {} id '{}'", kind, id),
            Self::MissingReference {
                kind,
                id,
                reference,
            } => write!(f, "{} '{}' references missing '{}'", kind, id, reference),
            Self::EmptyScope { kind, id } => write!(f, "{} '{}' has empty scope", kind, id),
            Self::InvalidSpec(msg) => write!(f, "invalid spec: {}", msg),
            Self::Message(msg) => write!(f, "{msg}"),
            Self::Poisoned(name) => write!(f, "shared state '{}' was poisoned", name),
        }
    }
}

impl Error for BraidError {}

impl From<String> for BraidError {
    fn from(value: String) -> Self {
        Self::Message(value)
    }
}

impl From<&str> for BraidError {
    fn from(value: &str) -> Self {
        Self::Message(value.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::BraidError;
    use crate::pipeline::{BufferSlot, KernelKind};

    #[test]
    fn error_display_covers_common_variants() {
        assert_eq!(BraidError::Cancelled.to_string(), "job cancelled");
        assert_eq!(BraidError::UnknownJob.to_string(), "unknown job");
        assert_eq!(
            BraidError::ExecutorShutdown.to_string(),
            "executor is shut down"
        );
        assert_eq!(
            BraidError::BackendRejectedKernel(KernelKind(7)).to_string(),
            "backend rejected kernel kind 7"
        );
        assert_eq!(
            BraidError::MissingBuffer(BufferSlot(4)).to_string(),
            "missing buffer at slot 4"
        );
        assert_eq!(
            BraidError::InvalidBufferType {
                slot: BufferSlot(9),
                expected: crate::pipeline::ElementKind::F32
            }
            .to_string(),
            "invalid buffer type at slot 9 expected F32"
        );
        assert_eq!(
            BraidError::DuplicateId {
                kind: "node",
                id: "a".to_owned()
            }
            .to_string(),
            "duplicate node id 'a'"
        );
        assert_eq!(
            BraidError::Poisoned("stack.state").to_string(),
            "shared state 'stack.state' was poisoned"
        );
        assert_eq!(BraidError::from("oops").to_string(), "oops");
    }

    #[test]
    fn from_string_and_str_construct_message_errors() {
        assert!(matches!(
            BraidError::from("hello".to_owned()),
            BraidError::Message(msg) if msg == "hello"
        ));
        assert!(matches!(
            BraidError::from("world"),
            BraidError::Message(msg) if msg == "world"
        ));
    }

    #[test]
    fn poisoned_constructor_preserves_label() {
        assert!(matches!(
            BraidError::poisoned("planner.state"),
            BraidError::Poisoned("planner.state")
        ));
    }
}
