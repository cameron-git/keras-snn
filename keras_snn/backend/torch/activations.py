import torch
import torch.utils.cpp_extension






def spike_fn(surrogate="sigmoid", **kwargs):
    """
    Returns a spiking activation function (Heaviside) with surrogate gradient based on the given surrogate.

    Args:
        surrogate (str): The surrogate function to use. Defaults to "sigmoid".

    Returns:
        function: The spike function.

    Raises:
        ValueError: If the given surrogate is unknown.
    """

    torch.utils.cpp_extension.load(
            name="warp_perspective",
            sources=["./csrc/op.cpp"],
            extra_ldflags=[],
            is_python_module=False,
            verbose=True
        )

    class sg(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.heaviside(x, torch.zeros_like(x))

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            return grad_input

    if surrogate == "sigmoid":
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]

        def sg_bwd(ctx, grad_output):
            (x,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            x = torch.sigmoid(x)
            grad_input = grad_input * x * (1 - x) * alpha
            return grad_input

    else:
        raise ValueError(f"Unknown surrogate: {surrogate}")

    sg.backward = sg_bwd

    return sg.apply
