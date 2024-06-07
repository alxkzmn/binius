// Copyright 2024 Ulvetanna Inc.

use super::{
	super::hasher::{HashDigest, Hasher},
	arch::Groestl256,
};
use crate::HasherDigest;
use binius_field::{
	underlier::WithUnderlier, BinaryField8b, ExtensionField, PackedBinaryField32x8b,
	PackedExtension, PackedField, PackedFieldIndexable,
};
use bytemuck::must_cast_slice_mut;
use digest::Digest;
use p3_symmetric::{CompressionFunction, PseudoCompressionFunction};
use std::{marker::PhantomData, slice};

pub type GroestlDigest = PackedBinaryField32x8b;

#[derive(Debug, Default, Clone)]
pub struct GroestlHasher<T> {
	inner: Groestl256,
	_t_marker: PhantomData<T>,
}

impl<P> Hasher<P> for GroestlHasher<P>
where
	P: PackedExtension<BinaryField8b, PackedSubfield: PackedFieldIndexable>,
	P::Scalar: ExtensionField<BinaryField8b>,
{
	type Digest = GroestlDigest;

	fn new() -> Self {
		Self {
			inner: Groestl256::new(),
			_t_marker: PhantomData,
		}
	}

	fn update(&mut self, data: impl AsRef<[P]>) {
		self.inner.update(to_u8_slice(data.as_ref()))
	}

	fn chain_update(self, data: impl AsRef<[P]>) -> Self {
		let Self { inner, _t_marker } = self;
		Self {
			inner: inner.chain_update(to_u8_slice(data.as_ref())),
			_t_marker,
		}
	}

	fn finalize(self) -> GroestlDigest {
		let mut digest = GroestlDigest::default();
		self.finalize_into(&mut digest);
		digest
	}

	fn finalize_into(self, out: &mut GroestlDigest) {
		let digest_bytes: &mut [u8] = must_cast_slice_mut(slice::from_mut(out));
		self.inner.finalize_into(digest_bytes.into())
	}

	fn finalize_reset(&mut self) -> Self::Digest {
		let mut digest = GroestlDigest::default();
		self.finalize_into_reset(&mut digest);
		digest
	}

	fn finalize_into_reset(&mut self, out: &mut Self::Digest) {
		let digest_bytes: &mut [u8] = must_cast_slice_mut(slice::from_mut(out));
		self.inner.finalize_into_reset(digest_bytes.into())
	}

	fn reset(&mut self) {
		self.inner.reset()
	}
}

fn to_u8_slice<
	PT: PackedField<Scalar: ExtensionField<BinaryField8b>>
		+ PackedExtension<BinaryField8b, PackedSubfield: PackedFieldIndexable>,
>(
	slice: &[PT],
) -> &[u8] {
	let packed_subfields = PT::cast_bases(slice);
	let scalars = PT::PackedSubfield::unpack_scalars(packed_subfields);
	BinaryField8b::to_underliers_ref(scalars)
}

#[derive(Debug, Default, Clone)]
pub struct GroestlDigestCompression;

impl PseudoCompressionFunction<GroestlDigest, 2> for GroestlDigestCompression {
	fn compress(&self, input: [GroestlDigest; 2]) -> GroestlDigest {
		HasherDigest::<GroestlDigest, GroestlHasher<GroestlDigest>>::hash(&input[..])
	}
}

impl CompressionFunction<GroestlDigest, 2> for GroestlDigestCompression {}

#[cfg(test)]
mod tests {
	use super::*;
	use hex_literal::hex;

	#[test]
	fn test_groestl_hash() {
		let expected = hex!("5bea5b2e398c903f0127a3467a961dd681069d06632502aa4297580b8ba50c75");
		let digest =
			GroestlDigestCompression.compress([GroestlDigest::default(), GroestlDigest::default()]);
		assert_eq!(to_u8_slice(&[digest]), &expected);
	}
}
