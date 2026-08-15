from unittest.mock import patch

from interactive_world_sim.algorithms.models.attention import Attention


def test_attention_construction_without_cuda() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "torch.cuda.get_device_properties",
            side_effect=AssertionError("CUDA properties must not be queried"),
        ),
    ):
        attention = Attention(query_dim=16)

    assert attention.cuda_backends
