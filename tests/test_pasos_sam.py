"""Contratos numéricos de SAM, restauración, BatchNorm, dropout y AMP."""
from copy import deepcopy
import unittest
import torch
from cancer_mama import experimento_sam as exp


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(3, 4)
        self.bn = torch.nn.BatchNorm1d(4)
        self.drop = torch.nn.Dropout(.4)
        self.out = torch.nn.Linear(4, 1)
        self.masks = []

    def forward(self, x, clinical=None):
        z = self.drop(self.bn(self.hidden(x)))
        self.masks.append((z != 0).detach().clone())
        return self.out(z)


class SAMStepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(314)
        self.x = torch.randn(8, 3)
        self.y = torch.tensor([0., 1.] * 4)
        self.device = torch.device('cpu')
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def scaler(self):
        return torch.amp.GradScaler('cuda', enabled=False)

    def test_sam_matches_analytic_gradient_on_quadratic(self):
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor([1., 2.]))
            def forward(self, x, clinical=None):
                return (x * self.w).sum(dim=1, keepdim=True)
        model = Quadratic()
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        # mean squared loss: f(w)=.5*(w1**2 + w2**2), grad=w.
        expected = model.w.detach() - .1 * (model.w.detach() + .05 * model.w.detach() / model.w.detach().norm())
        result = exp.optimization_step(model, opt, self.scaler(), torch.nn.MSELoss(),
                                       torch.eye(2), torch.zeros(2), None, self.device, False, .05, 1e6)
        torch.testing.assert_close(model.w, expected)
        self.assertGreater(result['perturbed_loss'], result['loss'])
        self.assertFalse(result['skipped'])

    def test_rho_zero_matches_original_adamw_update(self):
        model = Toy().train()
        reference = deepcopy(model)
        opt = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.001)
        refopt = torch.optim.AdamW(reference.parameters(), lr=.0008, weight_decay=.001)
        torch.manual_seed(42)
        self.criterion(reference(self.x).squeeze(1), self.y).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.)
        refopt.step()
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        exp.optimization_step(model, opt, self.scaler(), self.criterion, self.x, self.y,
                              None, self.device, False, 0)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_sam_restores_weights_before_adamw_and_counts_bn_once(self):
        model = Toy().train()
        reference = deepcopy(model)
        originals = [p.detach().clone() for p in model.parameters()]
        reference_gradients = []
        torch.manual_seed(42)
        self.criterion(reference(self.x).squeeze(1), self.y).backward()
        expected_rng = torch.get_rng_state()
        expected_bn = deepcopy(reference.bn.state_dict())
        opt = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.001)
        original_step = opt.step
        calls = []
        def checked_step(*args, **kwargs):
            for p, before in zip(model.parameters(), originals):
                torch.testing.assert_close(p, before, rtol=0, atol=0)
            reference_gradients.extend(p.grad.detach().clone() for p in model.parameters())
            calls.append(True)
            return original_step(*args, **kwargs)
        opt.step = checked_step
        torch.manual_seed(42)
        exp.optimization_step(model, opt, self.scaler(), self.criterion, self.x, self.y,
                              None, self.device, False, .05)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(model.masks), 2)
        self.assertTrue(torch.equal(*model.masks))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for name in ['running_mean', 'running_var', 'num_batches_tracked']:
            torch.testing.assert_close(model.bn.state_dict()[name], expected_bn[name], rtol=0, atol=0)
        # El optimizador realmente recibe el segundo gradiente: no el del control.
        self.assertTrue(any(not torch.equal(p.grad, g) for p, g in zip(reference.parameters(), reference_gradients)))

    def test_second_pass_exception_restores_parameters_and_bn(self):
        model = Toy().train()
        original = deepcopy(model.state_dict())
        reference = deepcopy(model)
        torch.manual_seed(42)
        reference(self.x)
        expected_rng = torch.get_rng_state()
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        calls = []
        def failing_loss(logits, y):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('forced second pass failure')
            return self.criterion(logits, y)
        torch.manual_seed(42)
        with self.assertRaisesRegex(RuntimeError, 'forced'):
            exp.optimization_step(model, opt, self.scaler(), failing_loss, self.x, self.y,
                                  None, self.device, False, .05)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, original[name], rtol=0, atol=0)
        for name in ['running_mean', 'running_var', 'num_batches_tracked']:
            torch.testing.assert_close(model.bn.state_dict()[name], reference.bn.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertFalse(opt.state)

    def test_cpu_nonfinite_gradient_rejected_and_negative_rho_rejected(self):
        model = Toy().train()
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        with self.assertRaises(ValueError):
            exp.optimization_step(model, opt, self.scaler(), self.criterion, self.x, self.y,
                                  None, self.device, False, -.1)
        hook = model.out.weight.register_hook(lambda g: torch.full_like(g, float('inf')))
        with self.assertRaisesRegex(FloatingPointError, 'Gradiente SAM'):
            exp.optimization_step(model, opt, self.scaler(), self.criterion, self.x, self.y,
                                  None, self.device, False, .05)
        hook.remove()
        self.assertFalse(opt.state)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for AMP overflow check')
    def test_amp_overflow_either_pass_skips_step_and_restores_weights(self):
        device = torch.device('cuda')
        exp.base.seed_everything(42)
        for overflow_pass in [1, 2]:
            with self.subTest(overflow_pass=overflow_pass):
                model = Toy().to(device).train()
                before = [p.detach().clone() for p in model.parameters()]
                opt = torch.optim.AdamW(model.parameters(), lr=.001)
                scaler = torch.amp.GradScaler('cuda', init_scale=128.)
                counter = []
                def inject(g):
                    counter.append(1)
                    return torch.full_like(g, float('inf')) if len(counter) == overflow_pass else g
                hook = model.out.weight.register_hook(inject)
                result = exp.optimization_step(model, opt, scaler, self.criterion,
                                               self.x.to(device), self.y.to(device), None, device, True, .05)
                self.assertTrue(result['skipped'])
                self.assertEqual(result['overflow_pass'], overflow_pass)
                self.assertEqual(scaler.get_scale(), 64.)
                self.assertFalse(opt.state)
                for p, old in zip(model.parameters(), before):
                    torch.testing.assert_close(p, old, rtol=0, atol=0)
                self.assertEqual(int(model.bn.num_batches_tracked), 1)
                hook.remove()
                # La escala sigue usable y el siguiente lote sí actualiza pesos.
                result = exp.optimization_step(model, opt, scaler, self.criterion,
                                               self.x.to(device), self.y.to(device), None, device, True, .05)
                self.assertFalse(result['skipped'])
                self.assertTrue(opt.state)


if __name__ == '__main__':
    unittest.main()
