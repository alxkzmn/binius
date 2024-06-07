// Copyright 2023 Ulvetanna Inc.
#![feature(step_trait)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]

pub mod aes_field;
pub mod affine_transformation;
pub mod arch;
pub mod arithmetic_traits;
pub mod as_packed_field;
pub mod binary_field;
mod binary_field_arithmetic;
pub mod error;
pub mod extension;
pub mod field;
mod macros;
pub mod packed;
pub mod packed_aes_field;
pub mod packed_binary_field;
pub mod packed_extension;
mod packed_polyval;
pub mod polyval;
pub mod transpose;
pub mod underlier;
pub mod util;

pub use aes_field::*;
pub use binary_field::*;
pub use error::*;
pub use extension::*;
pub use field::Field;
pub use packed::PackedField;
pub use packed_aes_field::*;
pub use packed_binary_field::*;
pub use packed_extension::*;
pub use polyval::*;
pub use transpose::{square_transpose, transpose_scalars, Error as TransposeError};
