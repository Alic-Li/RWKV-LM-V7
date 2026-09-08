"""CUDA regression tests for RWKV-7 state passing.

Run after installing the normal training requirements:
  RWKV_HEAD_SIZE=64 RWKV_MY_TESTING=x070 RWKV_TRAIN_TYPE=state pytest -q tests/test_state_passing.py
"""
import os
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.fixture(scope='module')
def ops():
    os.environ.setdefault('RWKV_HEAD_SIZE', '64')
    os.environ.setdefault('RWKV_MY_TESTING', 'x070')
    os.environ.setdefault('RWKV_TRAIN_TYPE', 'state')
    os.environ.setdefault('RWKV_JIT_ON', '0')
    from src.model import RWKV7_CLAMPW_CUDA, RWKV7_STATEPASS
    return RWKV7_CLAMPW_CUDA, RWKV7_STATEPASS


def test_zero_state_matches_clampw_and_gradients(ops):
    clampw, statepass = ops
    torch.manual_seed(123)
    B, T, H, N = 1, 32, 1, 64
    values = [torch.randn(B, T, H * N, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(6)]
    y_old = clampw(*values)
    y_old.float().square().mean().backward()
    grads_old = [x.grad.detach().clone() for x in values]

    values_new = [x.detach().clone().requires_grad_(True) for x in values]
    state = torch.zeros(B, H, N, N, device='cuda', dtype=torch.float32, requires_grad=True)
    y_new, state_out = statepass(state, *values_new)
    (y_new.float().square().mean() + state_out.square().mean()).backward()

    torch.testing.assert_close(y_new, y_old, rtol=2e-2, atol=2e-2)
    # state_out adds a deliberately different gradient contribution, so compare
    # gradients in a second output-only pass.
    values_out = [x.detach().clone().requires_grad_(True) for x in values]
    y_out, _ = statepass(torch.zeros_like(state), *values_out)
    y_out.float().square().mean().backward()
    for actual, expected in zip((x.grad for x in values_out), grads_old):
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    assert state.grad is not None and torch.count_nonzero(state.grad)


def test_chunked_wkv_equals_single_pass(ops):
    clampw, statepass = ops
    torch.manual_seed(456)
    B, T, H, N = 1, 64, 1, 64
    xs = [torch.randn(B, T, H * N, device='cuda', dtype=torch.bfloat16) for _ in range(6)]
    whole = clampw(*xs)
    state = torch.zeros(B, H, N, N, device='cuda', dtype=torch.float32)
    chunks = []
    for start in range(0, T, 16):
        y, state = statepass(state, *(x[:, start:start + 16] for x in xs))
        chunks.append(y)
    torch.testing.assert_close(torch.cat(chunks, dim=1), whole, rtol=2e-2, atol=2e-2)


def test_model_chunks_preserve_shift_states_and_time_state_gradient(ops):
    from src.model import RWKV
    args = SimpleNamespace(n_layer=2, n_embd=64, dim_att=64, dim_ffn=224,
                           head_size=64, vocab_size=128, ctx_len=32,
                           grad_cp=0, my_testing='x070', train_type='state',
                           chunk_ctx=16, state_detach=0)
    torch.manual_seed(789)
    model = RWKV(args).cuda().bfloat16().train()
    # State must remain fp32 at the kernel boundary even under bf16 model weights.
    for block in model.blocks:
        block.att.time_state.data = block.att.time_state.data.float()
        torch.nn.init.normal_(block.att.output.weight, std=0.02)
        torch.nn.init.normal_(block.ffn.value.weight, std=0.02)
    for name, param in model.named_parameters():
        param.requires_grad = name.endswith('att.time_state')
    idx = torch.randint(args.vocab_size, (1, 32), device='cuda')

    initial = model.init_trainable_state(1, idx.device)
    whole, whole_state = model(idx, initial)
    state = model.init_trainable_state(1, idx.device)
    pieces = []
    for start in range(0, 32, 16):
        logits, state = model(idx[:, start:start + 16], state)
        pieces.append(logits)
    torch.testing.assert_close(torch.cat(pieces, dim=1), whole, rtol=5e-2, atol=5e-2)
    for expected, actual in zip(whole_state, state):
        torch.testing.assert_close(actual.wkv_state, expected.wkv_state, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(actual.att_shift, expected.att_shift, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(actual.ffn_shift, expected.ffn_shift, rtol=5e-2, atol=5e-2)

    whole.float().square().mean().backward()
    assert all(name.endswith('att.time_state') for name, _ in [(n, p) for n, p in model.named_parameters() if p.requires_grad])
    assert all(block.att.time_state.grad is not None and torch.count_nonzero(block.att.time_state.grad) for block in model.blocks)
