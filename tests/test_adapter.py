import pytest

from kvbridge.adapter import UnsupportedAdapter


def test_default_adapter_refuses_cross_model_transfer():
    with pytest.raises(NotImplementedError):
        UnsupportedAdapter().adapt(None, None, None, "target")
