// Copyright 2024-2025 Irreducible Inc.

use binius_circuits::{
	builder::{types::U, ConstraintSystemBuilder},
	unconstrained::unconstrained,
};
use binius_core::{
	constraint_system::{self, ConstraintSystem, Proof},
	fiat_shamir::HasherChallenger,
	oracle::OracleId,
	tower::CanonicalTowerFamily,
	witness::MultilinearExtensionIndex,
};
use binius_field::{
	arch::OptimalUnderlier, as_packed_field::PackedType, BinaryField128b, BinaryField1b,
};
use binius_hal::{make_portable_backend, CpuBackend};
use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
use binius_utils::{
	checked_arithmetics::log2_ceil_usize, rayon::adjust_thread_pool, SerializeBytes,
};
use bytesize::ByteSize;
use clap::{value_parser, Parser};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use tracing_profile::init_tracing;

#[derive(Debug, Parser)]
pub struct Args {
	/// The number of compressions to verify.
	n_compressions: u32,
	/// The negative binary logarithm of the Reed–Solomon code rate.
	#[arg(long, default_value_t = 1, value_parser = value_parser!(u32).range(1..))]
	log_inv_rate: u32,
}

pub const COMPRESSION_LOG_LEN: usize = 5;
const BATCH_SIZE: usize = 256;

fn sha256_no_lookup_prepare<'a>(
	allocator: &'a bumpalo::Bump,
) -> (
	ConstraintSystem<BinaryField128b>,
	Args,
	MultilinearExtensionIndex<'a, PackedType<OptimalUnderlier, BinaryField128b>>,
	CpuBackend,
) {
	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	//let _guard = init_tracing().expect("failed to initialize tracing");

	let args = Args {
		n_compressions: 33,
		log_inv_rate: 1,
	};

	println!("Verifying {} sha256 compressions", args.n_compressions);

	let log_n_compressions = log2_ceil_usize(args.n_compressions as usize);

	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating trace").entered();
	let input: [OracleId; 16] = array_util::try_from_fn(|i| {
		unconstrained::<BinaryField1b>(&mut builder, i, log_n_compressions + COMPRESSION_LOG_LEN)
	})
	.unwrap();

	let _state_out = binius_circuits::sha256::sha256(
		&mut builder,
		input,
		log_n_compressions + COMPRESSION_LOG_LEN,
	)
	.unwrap();
	drop(trace_gen_scope);

	let witness = builder
		.take_witness()
		.expect("builder created with witness");

	let constraint_system = builder.build().unwrap();

	let backend = make_portable_backend();

	(constraint_system, args, witness, backend)
}

fn sha256_with_lookup_prepare<'a>(
	allocator: &'a bumpalo::Bump,
) -> (
	ConstraintSystem<BinaryField128b>,
	Args,
	MultilinearExtensionIndex<'a, PackedType<OptimalUnderlier, BinaryField128b>>,
	CpuBackend,
) {
	adjust_thread_pool()
		.as_ref()
		.expect("failed to init thread pool");

	let args = Args {
		n_compressions: 33,
		log_inv_rate: 1,
	};

	//let _guard = init_tracing().expect("failed to initialize tracing");

	println!("Verifying {} sha256 compressions with lookups", args.n_compressions);

	let log_n_compressions = log2_ceil_usize(args.n_compressions as usize);

	let mut builder = ConstraintSystemBuilder::new_with_witness(&allocator);

	let trace_gen_scope = tracing::info_span!("generating witness").entered();
	let input: [OracleId; 16] = array_util::try_from_fn(|i| {
		unconstrained::<BinaryField1b>(&mut builder, i, log_n_compressions + COMPRESSION_LOG_LEN)
	})
	.unwrap();

	let _state_out = binius_circuits::lasso::sha256(
		&mut builder,
		input,
		log_n_compressions + COMPRESSION_LOG_LEN,
	)
	.unwrap();
	drop(trace_gen_scope);

	let witness = builder
		.take_witness()
		.expect("builder created with witness");

	let constraint_system = builder.build().unwrap();

	let backend = make_portable_backend();

	(constraint_system, args, witness, backend)
}

fn prove<'a>(
	constraint_system: ConstraintSystem<binius_field::BinaryField128b>,
	args: Args,
	witness: MultilinearExtensionIndex<'a, PackedType<OptimalUnderlier, BinaryField128b>>,
	backend: CpuBackend,
) -> (ConstraintSystem<BinaryField128b>, Args, Proof) {
	const SECURITY_BITS: usize = 100;

	let proof =
		constraint_system::prove::<
			U,
			CanonicalTowerFamily,
			Groestl256,
			Groestl256ByteCompression,
			HasherChallenger<Groestl256>,
			_,
		>(&constraint_system, args.log_inv_rate as usize, SECURITY_BITS, &[], witness, &backend)
		.unwrap();

	println!("Proof size: {}", ByteSize::b(proof.get_proof_size() as u64));

	(constraint_system, args, proof)
}

fn verify(
	constraint_system: ConstraintSystem<binius_field::BinaryField128b>,
	args: Args,
	proof: Proof,
) {
	const SECURITY_BITS: usize = 100;

	constraint_system::verify::<
		U,
		CanonicalTowerFamily,
		Groestl256,
		Groestl256ByteCompression,
		HasherChallenger<Groestl256>,
	>(&constraint_system, args.log_inv_rate as usize, SECURITY_BITS, &[], proof)
	.unwrap();
}

fn sha256_no_lookup(c: &mut Criterion) {
	let mut group = c.benchmark_group("sha256_no_lookup");
	group.sample_size(10);
	let allocator = bumpalo::Bump::new();

	//group.throughput(Throughput::Elements(BATCH_SIZE as _));
	group.bench_function("sha256_no_lookup_prove", |bench| {
		bench.iter_batched(
			|| sha256_no_lookup_prepare(&allocator),
			|(constraint_system, args, witness, backend)| {
				prove(constraint_system, args, witness, backend);
			},
			BatchSize::SmallInput,
		);
	});

	group.bench_function("sha256_no_lookup_verify", |bench| {
		bench.iter_batched(
			|| {
				let (constraint_system, args, witness, backend) =
					sha256_no_lookup_prepare(&allocator);
				prove(constraint_system, args, witness, backend)
			},
			|(constraint_system, args, proof)| {
				verify(constraint_system, args, proof);
			},
			BatchSize::SmallInput,
		);
	});
	group.finish();
}

fn sha256_with_lookup(c: &mut Criterion) {
	let mut group = c.benchmark_group("sha256_with_lookup");
	group.sample_size(10);
	let allocator = bumpalo::Bump::new();

	//group.throughput(Throughput::Elements(BATCH_SIZE as _));
	group.bench_function("sha256_with_lookup_prove", |bench| {
		bench.iter_batched(
			|| sha256_with_lookup_prepare(&allocator),
			|(constraint_system, args, witness, backend)| {
				prove(constraint_system, args, witness, backend);
			},
			BatchSize::SmallInput,
		);
	});

	group.bench_function("sha256_with_lookup_verify", |bench| {
		bench.iter_batched(
			|| {
				let (constraint_system, args, witness, backend) =
					sha256_with_lookup_prepare(&allocator);
				prove(constraint_system, args, witness, backend)
			},
			|(constraint_system, args, proof)| {
				verify(constraint_system, args, proof);
			},
			BatchSize::SmallInput,
		);
	});
	group.finish();
}

criterion_main!(sha256);
criterion_group!(sha256, sha256_no_lookup, sha256_with_lookup);
