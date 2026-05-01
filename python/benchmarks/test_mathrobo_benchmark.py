import pytest

from test_support.python_impl_bench import (
    COMPARISON_CASE_NAMES,
    RUST_ONLY_CASE_NAMES,
    comparison_case,
    rust_only_case,
)


@pytest.mark.dev
@pytest.mark.parametrize("impl", ["mathrobors", "mathrobo"])
@pytest.mark.parametrize("case_name", COMPARISON_CASE_NAMES)
def test_mathrobo_python_benchmark(benchmark, case_name, impl):
    case = comparison_case(case_name)
    fn = case.rust_fn if impl == "mathrobors" else case.vendor_fn
    benchmark(fn)


@pytest.mark.dev
@pytest.mark.parametrize("case_name", RUST_ONLY_CASE_NAMES)
def test_mathrobors_buffer_benchmark(benchmark, case_name):
    case = rust_only_case(case_name)
    benchmark(case.rust_fn)
