"""CPU-only smoke tests for the AlexNet implementation."""

from pathlib import Path

import torch

from src import AlexNet, ReLU


def test_relu_matches_rectified_linear_unit():
    x = torch.tensor([-2.0, 0.0, 3.0])
    expected = torch.tensor([0.0, 0.0, 3.0])

    assert torch.equal(ReLU(x), expected)


def test_forward_shape_and_finite_logits():
    torch.manual_seed(0)
    model = AlexNet(num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 10)
    assert torch.isfinite(output).all()


def test_backward_reaches_trainable_parameters():
    torch.manual_seed(0)
    model = AlexNet(num_classes=10)
    model.train()
    x = torch.randn(1, 3, 224, 224)

    loss = model(x).square().mean()
    loss.backward()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)


def test_parameter_counter_matches_pytorch():
    model = AlexNet()
    expected = sum(parameter.numel() for parameter in model.parameters())

    assert model.params() == expected


def test_default_checkpoint_points_to_checkpoints_directory():
    checkpoint = Path(AlexNet.DEFAULT_WEIGHTS)

    assert checkpoint.parent.name == "checkpoints"
    assert checkpoint.name == "alexnet_imagenette.pth"
