//! Small value and error contracts shared by production crates.

use std::error;
use std::fmt;

/// Workspace-wide result type for expected production errors.
pub type Result<T> = std::result::Result<T, Error>;

/// An expected configuration or native-contract failure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A user- or environment-supplied configuration value is invalid.
    InvalidConfiguration {
        /// The configuration key or subsystem.
        context: &'static str,
        /// A stable, actionable diagnostic.
        message: String,
    },
    /// A production native component violated its documented ABI contract.
    NativeContract {
        /// The native component name.
        component: &'static str,
        /// A stable, actionable diagnostic.
        message: String,
    },
}

impl Error {
    /// Builds an invalid-configuration error without panicking.
    #[must_use]
    pub fn invalid_configuration(context: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidConfiguration {
            context,
            message: message.into(),
        }
    }

    /// Builds a native-contract error without exposing backend-specific types.
    #[must_use]
    pub fn native_contract(component: &'static str, message: impl Into<String>) -> Self {
        Self::NativeContract {
            component,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { context, message } => {
                write!(formatter, "invalid {context} configuration: {message}")
            }
            Self::NativeContract { component, message } => {
                write!(formatter, "{component} native contract error: {message}")
            }
        }
    }
}

impl error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn errors_have_actionable_context() {
        let error = Error::invalid_configuration("CUDA architecture", "expected digits");
        assert_eq!(
            error.to_string(),
            "invalid CUDA architecture configuration: expected digits"
        );
    }
}
