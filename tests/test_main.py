# tests/test_main.py

import torch

from src.helpers import fast_resize_m1_1


def test_fast_resize_m1_1():
    # Test with tensor in range [0, 1]
    tensor_0_1 = torch.tensor([[0.0, 0.5, 1.0]])
    result = fast_resize_m1_1(tensor_0_1)
    tensor_0_1 = tensor_0_1.unsqueeze(1).unsqueeze(2)  # Add batch and channel dimensions

    assert torch.allclose(result, torch.tensor([[[[-1.0, 0.0, 1.0]]]]))
    assert result.shape == tensor_0_1.shape

    # Test with tensor in range [0, 255]
    tensor_0_255 = torch.tensor([[0.0, 127.5, 255.0]])
    result = fast_resize_m1_1(tensor_0_255)
    tensor_0_255 = tensor_0_255.unsqueeze(1).unsqueeze(2)
    assert torch.allclose(result, torch.tensor([[[[-1.0, 0.0, 1.0]]]]))

    # Test with arbitrary range
    tensor_arbitrary = torch.tensor([[-30.0, 10.0, 20.0]])
    result = fast_resize_m1_1(tensor_arbitrary)
    tensor_arbitrary = tensor_arbitrary.unsqueeze(1).unsqueeze(2)
    assert torch.allclose(result, torch.tensor([[[[-1.0, 0.6, 1.0]]]]))

    # Test with single value (edge case)
    tensor_single = torch.tensor([[5.0]])
    result = fast_resize_m1_1(tensor_single)
    tensor_single = tensor_single.unsqueeze(1).unsqueeze(2)
    assert result.shape == torch.Size([1, 1, 1, 1])
