"""Exercise the production training loop without datasets or CLI side effects."""

import ast
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_training_loop():
    # Importing this entrypoint parses CLI arguments and imports dataset/FLOP
    # dependencies. Execute its actual training definitions without those effects.
    path = ROOT / "train_sat_solar_cast.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"EarlyStopping", "train_model"}
    ]
    namespace = {"__name__": "memory_training_test", "os": os, "torch": torch}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["train_model"]


class TinyPredictor(nn.Module):
    def __init__(self, use_memory):
        super().__init__()
        self.use_memory = use_memory
        self.satellite_memory = nn.Module()
        self.satellite_memory.memory_w = nn.Parameter(torch.tensor(0.5))
        self.context_scale = nn.Parameter(torch.tensor(0.8))
        self.matching_scale = nn.Parameter(torch.tensor(0.6))
        self.point_scale = nn.Parameter(torch.tensor(0.25))
        self.forward_flags = []
        self.phase = None

    def forward(self, x, y, multi_input, phase=1):
        self.phase = phase
        if self.training and torch.is_grad_enabled():
            self.forward_flags.append((phase, self.satellite_memory.memory_w.requires_grad))
        prediction = self.point_scale * x[:, :, 4:5, 0, 0]
        if self.use_memory:
            images = y if phase == 1 else x
            scale = self.context_scale if phase == 1 else self.matching_scale
            prediction = prediction + self.satellite_memory.memory_w * scale * images[:, :, 0:1, 0, 0]
        return prediction


class RecordingAdam(torch.optim.Adam):
    def __init__(self, model):
        super().__init__(model.parameters(), lr=0.01)
        self.model = model
        self.records = []

    def zero_grad(self, set_to_none=False):
        # Simulate the older zero-filled default: production must explicitly
        # request None to keep momentum from changing frozen parameters.
        super().zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        names = ["satellite_memory.memory_w", "matching_scale", "point_scale"]
        parameters = dict(self.model.named_parameters())
        before = {name: parameters[name].detach().clone() for name in names}
        gradients = {
            name: None if parameters[name].grad is None else parameters[name].grad.detach().clone()
            for name in names
        }
        result = super().step(closure)
        self.records.append({
            "phase": self.model.phase,
            "gradients": gradients,
            "changes": {name: float((parameters[name].detach() - before[name]).abs()) for name in names},
        })
        return result


class MemoryTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.train_model = staticmethod(load_training_loop())

    def run_training(self, use_memory=True):
        model = TinyPredictor(use_memory)
        optimizer = RecordingAdam(model)
        history = torch.ones(1, 2, 1, 1, 5)
        history[..., 4] = 0.2
        future = torch.ones_like(history)
        future[..., 4] = 0.0
        batch = (history, future, torch.zeros(1, 2, 4), torch.zeros(1, 2))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = SimpleNamespace(model_dir=directory, use_satellite_memory=use_memory)
            self.train_model(args, model, [batch] * 3, [batch], nn.MSELoss(), nn.MSELoss(),
                             optimizer, epochs=2, device=torch.device("cpu"))
        return model, optimizer.records

    def test_bank_updates_in_every_context_phase_across_batches_and_epochs(self):
        model, records = self.run_training()
        context = [row for row in records if row["phase"] == 1]
        self.assertEqual(len(context), 6)
        self.assertEqual([flag for phase, flag in model.forward_flags if phase == 1], [True] * 6)
        for row in context:
            gradient = row["gradients"]["satellite_memory.memory_w"]
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs()), 0)
            self.assertGreater(row["changes"]["satellite_memory.memory_w"], 0)

    def test_bank_is_frozen_in_every_matching_phase_despite_adam_momentum(self):
        model, records = self.run_training()
        matching = [row for row in records if row["phase"] == 2]
        self.assertEqual(len(matching), 6)
        self.assertEqual([flag for phase, flag in model.forward_flags if phase == 2], [False] * 6)
        for row in matching:
            self.assertIsNone(row["gradients"]["satellite_memory.memory_w"])
            self.assertEqual(row["changes"]["satellite_memory.memory_w"], 0)

    def test_matching_encoder_still_updates_while_bank_is_frozen(self):
        _, records = self.run_training()
        for row in records:
            if row["phase"] == 2:
                self.assertIsNotNone(row["gradients"]["matching_scale"])
                self.assertGreater(row["changes"]["matching_scale"], 0)

    def test_no_memory_path_uses_one_optimizer_step_per_batch(self):
        _, records = self.run_training(use_memory=False)
        self.assertEqual(len(records), 6)
        self.assertTrue(all(row["phase"] == 2 for row in records))
        self.assertTrue(all(row["changes"]["point_scale"] > 0 for row in records))
        self.assertTrue(all(row["gradients"]["satellite_memory.memory_w"] is None for row in records))


if __name__ == "__main__":
    unittest.main()
