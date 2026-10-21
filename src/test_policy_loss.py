import torch
from torch.utils.data import DataLoader, TensorDataset

import alphazero
from neural_net import NNArgs, NNWrapper
from train_offline_dataset import train_policy_only


def _wrapper(policy_loss):
    wrapper = NNWrapper.__new__(NNWrapper)
    wrapper.policy_loss = policy_loss
    return wrapper


def test_kl_policy_loss_reports_cross_entropy_minus_target_entropy():
    targets = torch.tensor([[0.25, 0.75, 0.0]], dtype=torch.float32)
    outputs = torch.log(torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.float32))

    ce = _wrapper("cross_entropy").loss_pi(targets, outputs)
    kl = _wrapper("kl").loss_pi(targets, outputs)
    entropy = -(targets[targets > 0] * torch.log(targets[targets > 0])).sum()

    assert torch.allclose(kl, ce - entropy)


def test_kl_policy_loss_has_same_output_gradient_as_soft_cross_entropy():
    targets = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    ce_outputs = torch.log(torch.tensor([[0.4, 0.4, 0.2]], dtype=torch.float32))
    kl_outputs = ce_outputs.clone()
    ce_outputs.requires_grad_(True)
    kl_outputs.requires_grad_(True)

    _wrapper("cross_entropy").loss_pi(targets, ce_outputs).backward()
    _wrapper("kl").loss_pi(targets, kl_outputs).backward()

    assert torch.allclose(kl_outputs.grad, ce_outputs.grad)


def test_orth_reg_lambda_tracks_training_term_when_enabled():
    args = NNArgs(
        num_channels=4,
        depth=1,
        kernel_size=3,
        dense_net=True,
        orth_reg_lambda=0.1,
    )
    wrapper = NNWrapper(alphazero.UltimateTicTacToeGS, args)
    canonical = torch.zeros(2, *alphazero.UltimateTicTacToeGS.CANONICAL_SHAPE())
    values = torch.zeros(2, alphazero.UltimateTicTacToeGS.NUM_PLAYERS() + 1)
    values[:, -1] = 1.0
    policies = torch.full(
        (2, alphazero.UltimateTicTacToeGS.NUM_MOVES()),
        1.0 / alphazero.UltimateTicTacToeGS.NUM_MOVES(),
    )
    dataloader = DataLoader(TensorDataset(canonical, values, policies), batch_size=2)

    class DummyRun:
        def __init__(self):
            self.tracks = []

        def track(self, value, name, **kwargs):
            self.tracks.append((name, value, kwargs))

    run = DummyRun()
    wrapper.train(dataloader, 1, run, epoch=0, total_train_steps=0, ema_averaging=False)

    orth_tracks = [value for name, value, _ in run.tracks if name == "orth_reg"]
    assert len(orth_tracks) == 1
    assert orth_tracks[0] > 0.0


def test_policy_only_training_records_policy_loss_without_value_loss():
    args = NNArgs(
        num_channels=4,
        depth=1,
        kernel_size=3,
        dense_net=True,
    )
    wrapper = NNWrapper(alphazero.UltimateTicTacToeGS, args)
    canonical = torch.zeros(4, *alphazero.UltimateTicTacToeGS.CANONICAL_SHAPE())
    values = torch.zeros(4, alphazero.UltimateTicTacToeGS.NUM_PLAYERS() + 1)
    policies = torch.zeros(4, alphazero.UltimateTicTacToeGS.NUM_MOVES())
    policies[:, 0] = 1.0
    dataloader = DataLoader(TensorDataset(canonical, values, policies), batch_size=2)

    class DummyRun:
        def __init__(self):
            self.tracks = []

        def track(self, value, name, **kwargs):
            self.tracks.append((name, value, kwargs))

    run = DummyRun()
    loss = train_policy_only(wrapper, dataloader, 1, run, epoch=0, total_train_steps=0)

    assert loss > 0.0
    loss_contexts = [
        kwargs.get("context", {}).get("type")
        for name, _value, kwargs in run.tracks
        if name == "loss"
    ]
    assert "policy" in loss_contexts
    assert "value" not in loss_contexts
