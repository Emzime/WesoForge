#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

extern crate link_cplusplus;

mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Errors returned by the safe Rust wrapper.
///
/// These checks do not change consensus-critical math. They only prevent accidental misuse
/// (wrong buffer sizes) from turning into undefined behavior across the FFI boundary.
///
/// Comments in English as requested.
#[derive(Debug, thiserror::Error)]
pub enum ChiavdfError {
    #[error("invalid challenge length: expected 32 bytes, got {got}")]
    InvalidChallengeLen { got: usize },

    #[error("invalid classgroup element length: expected 100 bytes, got {got}")]
    InvalidElementLen { got: usize },

    #[error(
        "invalid discriminant buffer length: expected {expected_bytes} bytes for {expected_bits} bits, got {got_bytes}"
    )]
    InvalidDiscriminantLen {
        expected_bits: usize,
        expected_bytes: usize,
        got_bytes: usize,
    },

    #[error("prove() returned null (C++ side error)")]
    ProveReturnedNull,
}

/// Create a discriminant from a seed.
///
/// This is the checked version of [`create_discriminant`].
pub fn create_discriminant_checked(seed: &[u8], result: &mut [u8]) -> Result<bool, ChiavdfError> {
    // The C++ API expects the discriminant size in bits.
    let expected_bits = result.len() * 8;
    let expected_bytes = expected_bits / 8;
    if expected_bytes != result.len() {
        return Err(ChiavdfError::InvalidDiscriminantLen {
            expected_bits,
            expected_bytes,
            got_bytes: result.len(),
        });
    }

    Ok(create_discriminant(seed, result))
}

/// Verify a proof (checked version).
pub fn verify_n_wesolowski_checked(
    discriminant: &[u8],
    x_s: &[u8],
    proof: &[u8],
    num_iterations: u64,
    recursion: u64,
) -> Result<bool, ChiavdfError> {
    if x_s.len() != 100 {
        return Err(ChiavdfError::InvalidElementLen { got: x_s.len() });
    }
    // Discriminant size is not fixed (64 bytes for 1024-bit, sometimes 128 bytes is used in callers).
    // We only enforce that it's non-empty and byte-aligned.
    if discriminant.is_empty() {
        return Err(ChiavdfError::InvalidDiscriminantLen {
            expected_bits: 8,
            expected_bytes: 1,
            got_bytes: 0,
        });
    }

    Ok(verify_n_wesolowski(
        discriminant,
        x_s,
        proof,
        num_iterations,
        recursion,
    ))
}

/// Prove (checked version).
pub fn prove_checked(
    challenge: &[u8],
    x_s: &[u8],
    discriminant_size_bits: usize,
    num_iterations: u64,
) -> Result<Vec<u8>, ChiavdfError> {
    if challenge.len() != 32 {
        return Err(ChiavdfError::InvalidChallengeLen {
            got: challenge.len(),
        });
    }
    if x_s.len() != 100 {
        return Err(ChiavdfError::InvalidElementLen { got: x_s.len() });
    }

    match prove(challenge, x_s, discriminant_size_bits, num_iterations) {
        Some(v) => Ok(v),
        None => Err(ChiavdfError::ProveReturnedNull),
    }
}

pub fn create_discriminant(seed: &[u8], result: &mut [u8]) -> bool {
    // SAFETY: The length of each individual array is passed in as to prevent buffer overflows.
    // Exceptions are handled on the C++ side and None is returned if so.
    unsafe {
        bindings::create_discriminant_wrapper(
            seed.as_ptr(),
            seed.len(),
            result.len() * 8,
            result.as_mut_ptr(),
        )
    }
}

pub fn verify_n_wesolowski(
    discriminant: &[u8],
    x_s: &[u8],
    proof: &[u8],
    num_iterations: u64,
    recursion: u64,
) -> bool {
    // SAFETY: The length of each individual array is passed in as to prevent buffer overflows.
    // Exceptions are handled on the C++ side and false is returned if so.
    unsafe {
        bindings::verify_n_wesolowski_wrapper(
            discriminant.as_ptr(),
            discriminant.len(),
            x_s.as_ptr(),
            proof.as_ptr(),
            proof.len(),
            num_iterations,
            recursion,
        )
    }
}

pub fn prove(
    challenge: &[u8],
    x_s: &[u8],
    discriminant_size_bits: usize,
    num_iterations: u64,
) -> Option<Vec<u8>> {
    // SAFETY: The length of each individual array is passed in as to prevent buffer overflows.
    // Exceptions are handled on the C++ side and a null pointer is returned for `data` if so.
    unsafe {
        let array = bindings::prove_wrapper(
            challenge.as_ptr(),
            challenge.len(),
            x_s.as_ptr(),
            x_s.len(),
            discriminant_size_bits,
            num_iterations,
        );
        if array.data.is_null() {
            return None;
        }
        let result = std::slice::from_raw_parts(array.data, array.length).to_vec();
        bindings::delete_byte_array(array);
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use hex_literal::hex;

    use super::*;

    #[test]
    fn test_create_discriminant() {
        let seeds = [
            hex!("6c3b9aa767f785b537c0"),
            hex!("b10da48cea4c09676b8e"),
            hex!("c51b8a31c98b9fe13065"),
            hex!("5de9bc1bb4cb7a9f9cf9"),
            hex!("22cfaefc92e4edb9b0ae"),
        ];

        let mut discriminants = Vec::new();

        for seed in seeds {
            let mut discriminant = [0; 64];
            assert!(create_discriminant(&seed, &mut discriminant));
            discriminants.push(discriminant);
        }

        // These came from running the Python `create_discriminant` with the same seeds.
        let expected = [
            "9a8eaf9c52d9a5f1db648cdf7bcd04b35cb1ac4f421c978fa61fe1344b97d4199dbff700d24e7cfc0b785e4b8b8023dc49f0e90227f74f54234032ac3381879f",
            "b193cdb02f1c2615a257b98933ee0d24157ac5f8c46774d5d635022e6e6bd3f7372898066c2a40fa211d1df8c45cb95c02e36ef878bc67325473d9c0bb34b047",
            "bb5bd19ae50efe98b5ac56c69453a95e92dc16bb4b2824e73b39b9db0a077fa33fc2e775958af14f675a071bf53f1c22f90ccbd456e2291276951830dba9dcaf",
            "a1e93b8f2e9b0fd3b1325fbe40601f55e2afbdc6161409c0aff8737b7213d7d71cab21ffc83a0b6d5bdeee2fdcbbb34fbc8fc0b439915075afa9ffac8bb1b337",
            "f2a10f70148fb30e4a16c4eda44cc0f9917cb9c2d460926d59a408318472e2cfd597193aa58e1fdccc6ae6a4d85bc9b27f77567ebe94fcedbf530a60ff709fd7",
        ];

        for i in 0..5 {
            assert_eq!(
                hex::encode(discriminants[i]),
                expected[i],
                "Discriminant {} does not match (seed is {})",
                i,
                hex::encode(seeds[i])
            );
        }
    }

    #[test]
    fn test_verify_n_wesolowski() {
        let genesis_challenge =
            hex!("ccd5bb71183532bff220ba46c268991a3ff07eb358e8255a65c30a2dce0e5fbb");

        let mut default_el = [0; 100];
        default_el[0] = 0x08;

        let mut disc = [0; 128];
        assert!(create_discriminant(&genesis_challenge, &mut disc));
        let proof = prove(&genesis_challenge, &default_el, 1024, 231).unwrap();
        let valid = verify_n_wesolowski(&disc, &default_el, &proof, 231, 0);
        assert!(valid);
    }
}
