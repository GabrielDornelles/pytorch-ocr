import pytest
from models.crnn import CRNN
import torch


@pytest.mark.parametrize("resolution", [(180, 50), (224, 224), (400, 400), (400, 100)])
@pytest.mark.parametrize("dims", [256, 1000])
@pytest.mark.parametrize("num_chars", [50])
@pytest.mark.parametrize("use_attention", [True])
@pytest.mark.parametrize("use_ctc", [True])
@pytest.mark.parametrize("grayscale", [True, False])
def test_init_and_forward(resolution, dims, num_chars, use_attention, use_ctc, grayscale):
    channels = 1 if grayscale else 3
    dummy_input = torch.zeros(1, channels, resolution[1], resolution[0])
    model = CRNN(
        resolution=resolution,
        dims=dims,
        num_chars=num_chars,
        use_attention=use_attention,
        use_ctc=use_ctc,
        grayscale=grayscale,
    )
    output, _ = model(dummy_input)
    assert len(output.shape) == 3
    assert output.shape[-1] == num_chars + 1
    assert output.shape[1] > 0  # sequence length must be bigger than 0


@pytest.mark.parametrize("resolution", [(180, 50), (224, 224), (400, 400), (400, 100)])
@pytest.mark.parametrize("dims", [256])
@pytest.mark.parametrize("num_chars", [50, 100])
@pytest.mark.parametrize("use_attention", [True])
@pytest.mark.parametrize("use_ctc", [True, False])
@pytest.mark.parametrize("grayscale", [True])
def test_losses_and_backward(resolution, dims, num_chars, use_attention, use_ctc, grayscale):
    channels = 1 if grayscale else 3
    dummy_input = torch.randn(2, channels, resolution[1], resolution[0])
    dummy_target = torch.randint(low=0, high=num_chars + 1, size=(2, 6), dtype=torch.long)
    model = CRNN(
        resolution=resolution,
        dims=dims,
        num_chars=num_chars,
        use_attention=use_attention,
        use_ctc=use_ctc,
        grayscale=grayscale,
    )
    _, loss = model(dummy_input, dummy_target)
    loss.backward()

    assert loss.item() >= 0

    found_grad = 0
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            found_grad += 1

    assert found_grad > 20  # clever way to check gradients existence
