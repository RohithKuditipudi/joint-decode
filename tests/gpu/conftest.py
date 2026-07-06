import pytest

pytest.importorskip("vllm", reason="GPU tests require the vllm engine")
