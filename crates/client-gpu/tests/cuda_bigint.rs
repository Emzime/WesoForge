use bbr_client_gpu::cuda::compute;

#[test]
fn cuda_bigint_selftest() {
    // Device 0 is the default for CI/dev; users can add variants later.
    compute::selftest_bigint_square_u32_32(0).expect("CUDA bigint square selftest failed");
}
