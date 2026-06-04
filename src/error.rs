use crate::pipeline::ElementKind;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub type BraidResult<T> = Result<T, BraidError>;

#[derive(Clone, Debug)]
pub enum BraidError {
    Cancelled,
    UnknownJob,
    ExecutorShutdown,
    BackendRejectedKernel(u32),
    MissingBuffer(u16),
    InvalidBufferType {
        slot: u16,
        expected: ElementKind,
    },
    DuplicateId {
        kind: &'static str,
        id: String,
    },
    MissingReference {
        kind: &'static str,
        id: String,
        reference: String,
    },
    EmptyScope {
        kind: &'static str,
        id: String,
    },
    InvalidSpec(String),
    Message(String),
    Poisoned(&'static str),
}

impl BraidError {
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
