import unittest
import numpy as np
from tinygrad import Tensor, Device, dtypes, nn, TinyJit, Context

DEV = "GL21"
# Ensure GL21 backend is initialized and patched
Device[DEV]

class TestGL21DeviceAndAllocator(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.device = Device[DEV]

  def test_device_init(self):
    self.assertIsNotNone(self.device)
    self.assertIsNotNone(self.device.allocator)
    self.assertIsNotNone(self.device.ctx)

  def test_dtypes_copyin_copyout(self):
    test_cases = [
      (dtypes.float32, [0.0, 1.0, -2.5, 3.14159, 100.0]),
      (dtypes.float64, [0.0, 1.0, -2.5, 3.14159, 100.0]),
      (dtypes.int32, [0, 1, -1, 42, -1000, 1000000]),
      (dtypes.uint32, [0, 1, 42, 1000, 1000000]),
      (dtypes.int8, [0, 1, -1, 42, -128, 127]),
      (dtypes.uint8, [0, 1, 42, 200, 255]),
      (dtypes.int16, [0, 1, -1, 42, -32768, 32767]),
      (dtypes.uint16, [0, 1, 42, 1000, 65535]),
      (dtypes.bool, [True, False, True, True, False]),
    ]
    for dt, data in test_cases:
      with self.subTest(dtype=dt.name):
        t = Tensor(data, dtype=dt, device=DEV)
        out = t.numpy()
        expected = np.array(data, dtype=np.dtype(dt.fmt))
        if dtypes.is_float(dt):
          np.testing.assert_allclose(out, expected, rtol=1e-6)
        else:
          np.testing.assert_array_equal(out, expected)

  def test_large_buffer_allocation(self):
    # Tests textures wrapping across max width of 8192
    sizes = [8192, 8193, 16384, 65536]
    for size in sizes:
      with self.subTest(size=size):
        data = np.arange(size, dtype=np.float32)
        t = Tensor(data, device=DEV)
        out = t.numpy()
        np.testing.assert_allclose(out, data)

  def test_small_and_empty_buffers(self):
    t1 = Tensor([42.0], device=DEV)
    np.testing.assert_allclose(t1.numpy(), [42.0])

  def test_read_write_aliasing(self):
    # Output buffer is also an input buffer (requires temp texture handling in GL21Program)
    x = Tensor.ones(10, device=DEV)
    for _ in range(5):
      x = (x + 1.0).realize()
    np.testing.assert_allclose(x.numpy(), np.full(10, 6.0))


class TestGL21MathAndALU(unittest.TestCase):
  def test_basic_arithmetic(self):
    a = Tensor([1.0, 2.0, 3.0, 4.0], device=DEV)
    b = Tensor([5.0, 6.0, 7.0, 8.0], device=DEV)
    np.testing.assert_allclose((a + b).numpy(), [6.0, 8.0, 10.0, 12.0])
    np.testing.assert_allclose((a - b).numpy(), [-4.0, -4.0, -4.0, -4.0])
    np.testing.assert_allclose((a * b).numpy(), [5.0, 12.0, 21.0, 32.0])
    np.testing.assert_allclose((b / a).numpy(), [5.0, 3.0, 7.0 / 3.0, 2.0], rtol=1e-5)
    np.testing.assert_allclose((-a).numpy(), [-1.0, -2.0, -3.0, -4.0])

  def test_pow_and_mod(self):
    a = Tensor([2.0, 3.0, 4.0], device=DEV)
    np.testing.assert_allclose((a ** 2).numpy(), [4.0, 9.0, 16.0])
    np.testing.assert_allclose((a ** 3).numpy(), [8.0, 27.0, 64.0])
    b = Tensor([10.0, 7.0, 15.0], device=DEV)
    m = Tensor([3.0, 4.0, 6.0], device=DEV)
    np.testing.assert_allclose((b % m).numpy(), [1.0, 3.0, 3.0])

  def test_math_functions(self):
    x = Tensor([0.1, 0.5, 1.0, 2.0], device=DEV)
    np.testing.assert_allclose(x.exp().numpy(), np.exp([0.1, 0.5, 1.0, 2.0]), rtol=1e-4)
    np.testing.assert_allclose(x.log().numpy(), np.log([0.1, 0.5, 1.0, 2.0]), rtol=1e-4)
    np.testing.assert_allclose(x.sin().numpy(), np.sin([0.1, 0.5, 1.0, 2.0]), rtol=1e-4)
    np.testing.assert_allclose(x.cos().numpy(), np.cos([0.1, 0.5, 1.0, 2.0]), rtol=1e-4)
    np.testing.assert_allclose(x.sqrt().numpy(), np.sqrt([0.1, 0.5, 1.0, 2.0]), rtol=1e-4)
    np.testing.assert_allclose(x.reciprocal().numpy(), 1.0 / np.array([0.1, 0.5, 1.0, 2.0]), rtol=1e-4)

  def test_min_max_clip_abs(self):
    a = Tensor([1.0, 5.0, -3.0, -0.5], device=DEV)
    b = Tensor([2.0, 3.0, 0.0, -1.0], device=DEV)
    np.testing.assert_allclose(a.maximum(b).numpy(), [2.0, 5.0, 0.0, -0.5])
    np.testing.assert_allclose(a.minimum(b).numpy(), [1.0, 3.0, -3.0, -1.0])
    np.testing.assert_allclose(a.abs().numpy(), [1.0, 5.0, 3.0, 0.5])
    np.testing.assert_allclose(a.clip(0.0, 4.0).numpy(), [1.0, 4.0, 0.0, 0.0])

  def test_comparisons(self):
    a = Tensor([1.0, 5.0, 3.0, 4.0], device=DEV)
    b = Tensor([2.0, 4.0, 3.0, 1.0], device=DEV)
    np.testing.assert_array_equal((a < b).numpy(), [True, False, False, False])
    np.testing.assert_array_equal((a <= b).numpy(), [True, False, True, False])
    np.testing.assert_array_equal((a > b).numpy(), [False, True, False, True])
    np.testing.assert_array_equal((a >= b).numpy(), [False, True, True, True])
    np.testing.assert_array_equal((a == b).numpy(), [False, False, True, False])
    np.testing.assert_array_equal((a != b).numpy(), [True, True, False, True])

  def test_where_ternary(self):
    cond = Tensor([True, False, True, False], device=DEV)
    x = Tensor([1.0, 2.0, 3.0, 4.0], device=DEV)
    y = Tensor([10.0, 20.0, 30.0, 40.0], device=DEV)
    res = cond.where(x, y).numpy()
    np.testing.assert_allclose(res, [1.0, 20.0, 3.0, 40.0])

  def test_bitwise_ops(self):
    a = Tensor([5, 12, 255, 0, 17], dtype=dtypes.int32, device=DEV)
    b = Tensor([3, 10, 15, 7, 31], dtype=dtypes.int32, device=DEV)
    np_a = np.array([5, 12, 255, 0, 17], dtype=np.int32)
    np_b = np.array([3, 10, 15, 7, 31], dtype=np.int32)
    np.testing.assert_array_equal((a & b).numpy(), np_a & np_b)
    np.testing.assert_array_equal((a | b).numpy(), np_a | np_b)
    np.testing.assert_array_equal((a ^ b).numpy(), np_a ^ np_b)
    np.testing.assert_array_equal((a << 2).numpy(), np_a << 2)
    np.testing.assert_array_equal((a >> 1).numpy(), np_a >> 1)


class TestGL21Casting(unittest.TestCase):
  def test_cast_ops(self):
    x = Tensor([1.2, 2.7, -3.8], device=DEV)
    xi = x.cast(dtypes.int32).numpy()
    np.testing.assert_array_equal(xi, np.array([1, 2, -3], dtype=np.int32))

    xb = x.cast(dtypes.bool).numpy()
    np.testing.assert_array_equal(xb, np.array([True, True, True]))

    t_int = Tensor([0, 1, 5], dtype=dtypes.int32, device=DEV)
    t_flt = t_int.cast(dtypes.float32).numpy()
    np.testing.assert_allclose(t_flt, [0.0, 1.0, 5.0])


class TestGL21MovementOps(unittest.TestCase):
  def test_reshape(self):
    t = Tensor.arange(24).to(DEV).reshape(2, 3, 4)
    self.assertEqual(t.shape, (2, 3, 4))
    np.testing.assert_array_equal(t.numpy(), np.arange(24).reshape(2, 3, 4))

  def test_permute(self):
    t = Tensor.arange(24).to(DEV).reshape(2, 3, 4).permute(2, 0, 1)
    self.assertEqual(t.shape, (4, 2, 3))
    np.testing.assert_array_equal(t.numpy(), np.arange(24).reshape(2, 3, 4).transpose(2, 0, 1))

  def test_expand(self):
    t = Tensor([1.0, 2.0, 3.0], device=DEV).unsqueeze(0).expand(4, 3)
    self.assertEqual(t.shape, (4, 3))
    expected = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
    np.testing.assert_allclose(t.numpy(), expected)

  def test_pad(self):
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], device=DEV)
    padded = t.pad(((1, 2), (2, 1)), value=0.0)
    expected = np.pad(np.array([[1.0, 2.0], [3.0, 4.0]]), ((1, 2), (2, 1)))
    np.testing.assert_allclose(padded.numpy(), expected)

  def test_shrink_slice(self):
    t = Tensor.arange(16).to(DEV).reshape(4, 4)
    shrunk = t.shrink(((1, 3), (1, 3)))
    expected = np.arange(16).reshape(4, 4)[1:3, 1:3]
    np.testing.assert_array_equal(shrunk.numpy(), expected)

  def test_flip(self):
    t = Tensor.arange(12).to(DEV).reshape(3, 4)
    np.testing.assert_array_equal(t.flip(0).numpy(), np.flip(np.arange(12).reshape(3, 4), axis=0))
    np.testing.assert_array_equal(t.flip(1).numpy(), np.flip(np.arange(12).reshape(3, 4), axis=1))

  def test_cat_and_stack(self):
    t1 = Tensor([[1, 2], [3, 4]], device=DEV)
    t2 = Tensor([[5, 6], [7, 8]], device=DEV)
    cat0 = Tensor.cat(t1, t2, dim=0).numpy()
    np.testing.assert_array_equal(cat0, [[1, 2], [3, 4], [5, 6], [7, 8]])
    cat1 = Tensor.cat(t1, t2, dim=1).numpy()
    np.testing.assert_array_equal(cat1, [[1, 2, 5, 6], [3, 4, 7, 8]])
    stk0 = Tensor.stack([t1, t2], dim=0).numpy()
    np.testing.assert_array_equal(stk0, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

  def test_multidim_slicing(self):
    grid = Tensor.arange(25).to(DEV).reshape(5, 5)
    np.testing.assert_array_equal(grid[1:4, 2:5].numpy(), np.arange(25).reshape(5, 5)[1:4, 2:5])
    np.testing.assert_array_equal(grid[::2, ::2].numpy(), np.arange(25).reshape(5, 5)[::2, ::2])


class TestGL21Reductions(unittest.TestCase):
  def test_global_reductions(self):
    data = np.random.randn(8, 8).astype(np.float32)
    t = Tensor(data, device=DEV)
    np.testing.assert_allclose(t.sum().numpy(), data.sum(), rtol=1e-4)
    np.testing.assert_allclose(t.max().numpy(), data.max(), rtol=1e-4)
    np.testing.assert_allclose(t.mean().numpy(), data.mean(), rtol=1e-4)

  def test_axis_reductions(self):
    data = np.random.randn(4, 5, 6).astype(np.float32)
    t = Tensor(data, device=DEV)
    for axis in [0, 1, 2, -1]:
      with self.subTest(axis=axis):
        np.testing.assert_allclose(t.sum(axis=axis).numpy(), data.sum(axis=axis), rtol=1e-4)
        np.testing.assert_allclose(t.max(axis=axis).numpy(), data.max(axis=axis), rtol=1e-4)
        np.testing.assert_allclose(t.mean(axis=axis).numpy(), data.mean(axis=axis), rtol=1e-4)

  def test_multiaxis_reductions(self):
    data = np.random.randn(3, 4, 5).astype(np.float32)
    t = Tensor(data, device=DEV)
    np.testing.assert_allclose(t.sum(axis=(0, 2)).numpy(), data.sum(axis=(0, 2)), rtol=1e-4)
    np.testing.assert_allclose(t.max(axis=(1, 2)).numpy(), data.max(axis=(1, 2)), rtol=1e-4)

  def test_keepdim_reductions(self):
    data = np.random.randn(3, 4).astype(np.float32)
    t = Tensor(data, device=DEV)
    s = t.sum(axis=1, keepdim=True)
    self.assertEqual(s.shape, (3, 1))
    np.testing.assert_allclose(s.numpy(), data.sum(axis=1, keepdims=True), rtol=1e-4)

  def test_argmax(self):
    data = np.array([[1.0, 5.0, 3.0], [8.0, 2.0, 4.0]], dtype=np.float32)
    t = Tensor(data, device=DEV)
    np.testing.assert_array_equal(t.argmax(axis=-1).numpy(), [1, 0])


class TestGL21MatmulAndLinear(unittest.TestCase):
  def test_matmul_square(self):
    for n in [2, 8, 32]:
      with self.subTest(n=n):
        a = np.random.randn(n, n).astype(np.float32)
        b = np.random.randn(n, n).astype(np.float32)
        t_a = Tensor(a, device=DEV)
        t_b = Tensor(b, device=DEV)
        out = (t_a @ t_b).numpy()
        expected = a @ b
        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)

  def test_matmul_rectangular(self):
    shapes = [(4, 8, 6), (1, 64, 32), (32, 16, 1)]
    for m, k, n in shapes:
      with self.subTest(m=m, k=k, n=n):
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        t_a = Tensor(a, device=DEV)
        t_b = Tensor(b, device=DEV)
        out = (t_a @ t_b).numpy()
        expected = a @ b
        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)

  def test_batched_matmul(self):
    a = np.random.randn(2, 4, 8).astype(np.float32)
    b = np.random.randn(2, 8, 6).astype(np.float32)
    t_a = Tensor(a, device=DEV)
    t_b = Tensor(b, device=DEV)
    out = (t_a @ t_b).numpy()
    expected = a @ b
    np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)

  def test_nn_linear(self):
    lin = nn.Linear(16, 8)
    lin.weight = lin.weight.to(DEV)
    if lin.bias is not None: lin.bias = lin.bias.to(DEV)
    inp = Tensor.randn(4, 16, device=DEV)
    out = lin(inp)
    self.assertEqual(out.shape, (4, 8))
    expected = inp.numpy() @ lin.weight.numpy().T + lin.bias.numpy()
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-3, atol=1e-3)


class TestGL21CNNAndPooling(unittest.TestCase):
  def test_conv2d_standard(self):
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    conv.weight = conv.weight.to(DEV)
    if conv.bias is not None: conv.bias = conv.bias.to(DEV)
    inp = Tensor.randn(2, 3, 16, 16, device=DEV)
    out = conv(inp)
    self.assertEqual(out.shape, (2, 8, 16, 16))

  def test_conv2d_stride_and_padding(self):
    conv = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
    conv.weight = conv.weight.to(DEV)
    if conv.bias is not None: conv.bias = conv.bias.to(DEV)
    inp = Tensor.randn(1, 4, 16, 16, device=DEV)
    out = conv(inp)
    self.assertEqual(out.shape, (1, 8, 8, 8))

  def test_conv2d_1x1(self):
    conv = nn.Conv2d(8, 16, kernel_size=1)
    conv.weight = conv.weight.to(DEV)
    if conv.bias is not None: conv.bias = conv.bias.to(DEV)
    inp = Tensor.randn(2, 8, 8, 8, device=DEV)
    out = conv(inp)
    self.assertEqual(out.shape, (2, 16, 8, 8))

  def test_max_pool2d(self):
    inp = Tensor.randn(2, 4, 16, 16, device=DEV)
    out = inp.max_pool2d(kernel_size=(2, 2))
    self.assertEqual(out.shape, (2, 4, 8, 8))

  def test_avg_pool2d(self):
    inp = Tensor.randn(2, 4, 16, 16, device=DEV)
    out = inp.avg_pool2d(kernel_size=(2, 2))
    self.assertEqual(out.shape, (2, 4, 8, 8))


class TestGL21ActivationsAndLosses(unittest.TestCase):
  def test_activations(self):
    x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x = Tensor(x_np, device=DEV)
    np.testing.assert_allclose(x.relu().numpy(), np.maximum(0, x_np))
    np.testing.assert_allclose(x.sigmoid().numpy(), 1.0 / (1.0 + np.exp(-x_np)), rtol=1e-4)
    np.testing.assert_allclose(x.tanh().numpy(), np.tanh(x_np), rtol=1e-4)

  def test_softmax_and_log_softmax(self):
    data = np.random.randn(3, 5).astype(np.float32)
    t = Tensor(data, device=DEV)
    # Softmax
    sm = t.softmax(axis=-1).numpy()
    np_exp = np.exp(data - data.max(axis=-1, keepdims=True))
    expected_sm = np_exp / np_exp.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(sm, expected_sm, rtol=1e-4)
    # LogSoftmax
    lsm = t.log_softmax(axis=-1).numpy()
    expected_lsm = np.log(expected_sm)
    np.testing.assert_allclose(lsm, expected_lsm, rtol=1e-4)

  def test_cross_entropy(self):
    preds = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], device=DEV)
    targets = Tensor([0, 1], device=DEV)
    loss = preds.cross_entropy(targets).numpy()
    self.assertFalse(np.isnan(loss))
    self.assertGreater(loss, 0.0)


class TestGL21AutogradAndBackprop(unittest.TestCase):
  def test_polynomial_gradient(self):
    # f(x) = 3*x^2 + 2*x + 1  => f'(x) = 6*x + 2
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = Tensor(x_data, device=DEV).realize()
    loss = (3 * (x ** 2) + 2 * x + 1).sum()
    loss.backward()
    self.assertIsNotNone(x.grad)
    expected_grad = 6 * x_data + 2
    np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-4)

  def test_linear_layer_backward(self):
    lin = nn.Linear(4, 2)
    lin.weight = lin.weight.to(DEV)
    if lin.bias is not None: lin.bias = lin.bias.to(DEV)
    x = Tensor.randn(3, 4, device=DEV).realize()
    target = Tensor.randn(3, 2, device=DEV).realize()
    pred = lin(x)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    self.assertIsNotNone(lin.weight.grad)
    self.assertEqual(lin.weight.grad.shape, (2, 4))
    self.assertFalse(np.isnan(lin.weight.grad.numpy()).any())
    if lin.bias is not None:
      self.assertIsNotNone(lin.bias.grad)
      self.assertEqual(lin.bias.grad.shape, (2,))
      self.assertFalse(np.isnan(lin.bias.grad.numpy()).any())

  def test_conv2d_backward(self):
    conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)
    conv.weight = conv.weight.to(DEV)
    if conv.bias is not None: conv.bias = conv.bias.to(DEV)
    x = Tensor.randn(1, 2, 8, 8, device=DEV).realize()
    out = conv(x)
    loss = (out ** 2).sum()
    loss.backward()
    self.assertIsNotNone(conv.weight.grad)
    self.assertFalse(np.isnan(conv.weight.grad.numpy()).any())

  def test_optimization_step(self):
    lin = nn.Linear(4, 2)
    lin.weight = lin.weight.to(DEV)
    if lin.bias is not None: lin.bias = lin.bias.to(DEV)
    x = Tensor.randn(4, 4, device=DEV).realize()
    target = Tensor.randn(4, 2, device=DEV).realize()
    initial_loss = float(((lin(x) - target) ** 2).mean().numpy())

    # 3 steps of SGD
    lr = 0.05
    for _ in range(3):
      loss = ((lin(x) - target) ** 2).mean()
      loss.backward()
      lin.weight = (lin.weight - lr * lin.weight.grad).realize()
      if lin.bias is not None:
        lin.bias = (lin.bias - lr * lin.bias.grad).realize()

    final_loss = float(((lin(x) - target) ** 2).mean().numpy())
    self.assertLess(final_loss, initial_loss)


class TestGL21NormalizationLayers(unittest.TestCase):
  def test_batchnorm2d_eval(self):
    bn = nn.BatchNorm2d(4)
    bn.weight = bn.weight.to(DEV)
    bn.bias = bn.bias.to(DEV)
    bn.running_mean = bn.running_mean.to(DEV)
    bn.running_var = bn.running_var.to(DEV)
    inp = Tensor.randn(2, 4, 8, 8, device=DEV)
    with Context(TRAINING=False):
      out = bn(inp)
      self.assertEqual(out.shape, (2, 4, 8, 8))
      self.assertFalse(np.isnan(out.numpy()).any())

  def test_layernorm(self):
    ln = nn.LayerNorm(16)
    if ln.weight is not None: ln.weight = ln.weight.to(DEV)
    if ln.bias is not None: ln.bias = ln.bias.to(DEV)
    inp = Tensor.randn(4, 8, 16, device=DEV)
    out = ln(inp)
    self.assertEqual(out.shape, (4, 8, 16))
    self.assertFalse(np.isnan(out.numpy()).any())

  def test_embedding(self):
    emb = nn.Embedding(10, 16)
    emb.weight = emb.weight.to(DEV)
    idx = Tensor([1, 4, 7, 2], device=DEV)
    out = emb(idx)
    self.assertEqual(out.shape, (4, 16))
    expected = emb.weight.numpy()[[1, 4, 7, 2]]
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4)


class TestGL21RandomAndInit(unittest.TestCase):
  def test_zeros_ones_full(self):
    z = Tensor.zeros(3, 4, device=DEV)
    np.testing.assert_array_equal(z.numpy(), np.zeros((3, 4), dtype=np.float32))
    o = Tensor.ones(3, 4, device=DEV)
    np.testing.assert_array_equal(o.numpy(), np.ones((3, 4), dtype=np.float32))
    f = Tensor.full((2, 3), 7.5, device=DEV)
    np.testing.assert_allclose(f.numpy(), np.full((2, 3), 7.5, dtype=np.float32))

  def test_arange(self):
    a = Tensor.arange(10).to(DEV)
    np.testing.assert_array_equal(a.numpy(), np.arange(10))

  def test_rand_randn_uniform_normal(self):
    r = Tensor.rand(10, 10, device=DEV).numpy()
    self.assertTrue((r >= 0.0).all() and (r <= 1.0).all())
    rn = Tensor.randn(10, 10, device=DEV).numpy()
    self.assertFalse(np.isnan(rn).any())
    u = Tensor.uniform(10, 10, low=-5.0, high=5.0, device=DEV).numpy()
    self.assertTrue((u >= -5.0).all() and (u <= 5.0).all())
    n = Tensor.normal(10, 10, mean=10.0, std=2.0, device=DEV).numpy()
    self.assertFalse(np.isnan(n).any())


class TestGL21JIT(unittest.TestCase):
  def test_jit_elementwise(self):
    @TinyJit
    def step(a, b):
      return ((a + b) * a - b).realize()

    t1 = Tensor([1.0, 2.0, 3.0], device=DEV)
    t2 = Tensor([4.0, 5.0, 6.0], device=DEV)
    for _ in range(3):
      res = step(t1, t2).numpy()
    expected = (np.array([1.0, 2.0, 3.0]) + np.array([4.0, 5.0, 6.0])) * np.array([1.0, 2.0, 3.0]) - np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(res, expected)

  def test_jit_model_step(self):
    lin = nn.Linear(8, 4)
    lin.weight = lin.weight.to(DEV)
    if lin.bias is not None: lin.bias = lin.bias.to(DEV)

    @TinyJit
    def forward(x):
      return lin(x).relu().realize()

    inp = Tensor.randn(2, 8, device=DEV)
    for _ in range(3):
      out = forward(inp).numpy()
    self.assertEqual(out.shape, (2, 4))
    self.assertFalse(np.isnan(out).any())


class TestGL21EndToEnd(unittest.TestCase):
  def test_mlp(self):
    class MLP:
      def __init__(self):
        self.l1 = nn.Linear(16, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 4)
        for l in [self.l1, self.l2, self.l3]:
          l.weight = l.weight.to(DEV)
          if l.bias is not None: l.bias = l.bias.to(DEV)

      def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        return self.l3(x)

    model = MLP()
    inp = Tensor.randn(4, 16, device=DEV)
    out = model(inp)
    self.assertEqual(out.shape, (4, 4))
    self.assertFalse(np.isnan(out.numpy()).any())

  def test_convnet(self):
    class ConvNet:
      def __init__(self):
        self.c1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 4 * 4, 10)
        for c in [self.c1, self.c2]:
          c.weight = c.weight.to(DEV)
          if c.bias is not None: c.bias = c.bias.to(DEV)
        self.fc.weight = self.fc.weight.to(DEV)
        if self.fc.bias is not None: self.fc.bias = self.fc.bias.to(DEV)

      def __call__(self, x):
        x = self.c1(x).relu().max_pool2d((2, 2))
        x = self.c2(x).relu().max_pool2d((2, 2))
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    model = ConvNet()
    inp = Tensor.randn(2, 3, 16, 16, device=DEV)
    out = model(inp)
    self.assertEqual(out.shape, (2, 10))
    self.assertFalse(np.isnan(out.numpy()).any())

  def test_residual_block(self):
    class ResBlock:
      def __init__(self, channels):
        self.c1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        for c in [self.c1, self.c2]:
          c.weight = c.weight.to(DEV)
          if c.bias is not None: c.bias = c.bias.to(DEV)

      def __call__(self, x):
        residual = x
        out = self.c1(x).relu()
        out = self.c2(out)
        return (out + residual).relu()

    block = ResBlock(channels=8)
    inp = Tensor.randn(2, 8, 16, 16, device=DEV)
    out = block(inp)
    self.assertEqual(out.shape, (2, 8, 16, 16))
    self.assertFalse(np.isnan(out.numpy()).any())

  def test_attention_block(self):
    class SelfAttention:
      def __init__(self, dim, heads):
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        for l in [self.wq, self.wk, self.wv, self.wo]:
          l.weight = l.weight.to(DEV)
          if l.bias is not None: l.bias = l.bias.to(DEV)

      def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        # Scaled dot-product attention
        scores = (q @ k.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attn = scores.softmax(axis=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)

    attn = SelfAttention(dim=32, heads=4)
    inp = Tensor.randn(2, 8, 32, device=DEV)
    out = attn(inp)
    self.assertEqual(out.shape, (2, 8, 32))
    self.assertFalse(np.isnan(out.numpy()).any())

  def test_transformer_block(self):
    class TransformerBlock:
      def __init__(self, dim, heads):
        self.attn = TestGL21EndToEnd._make_attn(dim, heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn1 = nn.Linear(dim, dim * 4)
        self.ffn2 = nn.Linear(dim * 4, dim)
        for l in [self.ln1, self.ln2, self.ffn1, self.ffn2]:
          if hasattr(l, 'weight') and l.weight is not None: l.weight = l.weight.to(DEV)
          if hasattr(l, 'bias') and l.bias is not None: l.bias = l.bias.to(DEV)

      def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn2(self.ffn1(self.ln2(x)).gelu())
        return x

    def _make_attn(dim, heads):
      class SA:
        def __init__(self):
          self.qkv = nn.Linear(dim, dim * 3)
          self.qkv.weight = self.qkv.weight.to(DEV)
          if self.qkv.bias is not None: self.qkv.bias = self.qkv.bias.to(DEV)
          self.proj = nn.Linear(dim, dim)
          self.proj.weight = self.proj.weight.to(DEV)
          if self.proj.bias is not None: self.proj.bias = self.proj.bias.to(DEV)
        def __call__(self, x):
          B, T, C = x.shape
          qkv = self.qkv(x).reshape(B, T, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
          q, k, v = qkv[0], qkv[1], qkv[2]
          scores = (q @ k.permute(0, 1, 3, 2)) / ((dim // heads) ** 0.5)
          attn = scores.softmax(axis=-1)
          out = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, C)
          return self.proj(out)
      return SA()

    TestGL21EndToEnd._make_attn = staticmethod(_make_attn)
    block = TransformerBlock(dim=32, heads=4)
    inp = Tensor.randn(2, 8, 32, device=DEV)
    out = block(inp)
    self.assertEqual(out.shape, (2, 8, 32))
    self.assertFalse(np.isnan(out.numpy()).any())

  def test_nn_mnist_linear_training_1to1_parity(self):
    np.random.seed(12345)
    w1 = np.random.randn(784, 128).astype(np.float32) * 0.05
    w2 = np.random.randn(128, 10).astype(np.float32) * 0.05
    x_val = np.random.rand(4, 1, 28, 28).astype(np.float32)
    y_val = np.array([2, 4, 3, 7])

    results = {}
    for dev in ['CPU', 'GL21']:
      l1 = Tensor(w1, device=dev)
      l2 = Tensor(w2, device=dev)
      optim = nn.optim.Adam([l1, l2], lr=0.001)
      x = Tensor(x_val, device=dev)
      y = Tensor(y_val, device=dev)

      losses = []
      with Context(TRAINING=1):
        for _ in range(10):
          optim.zero_grad()
          out = x.flatten(1).dot(l1).relu().dot(l2)
          loss = out.sparse_categorical_crossentropy(y).backward()
          optim.step()
          losses.append(loss.item())
      results[dev] = losses

    max_diff = max(abs(c - g) for c, g in zip(results['CPU'], results['GL21']))
    np.testing.assert_allclose(results['GL21'], results['CPU'], rtol=1e-4, atol=1e-4)
    self.assertLess(max_diff, 1e-4)

if __name__ == '__main__':
  unittest.main()
