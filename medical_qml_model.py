"""
Core logic for the Hybrid Deep Learning + Quantum ML medical-imaging demo.
No Streamlit imports here -- this module is pure model code so it can be
tested/reused independently of the UI.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

IMG_SIZE = 32
N_QUBITS = 4
N_QLAYERS = 2

# ---------------------------------------------------------------------------
# Synthetic "scan" data
# ---------------------------------------------------------------------------

def tissue_background(size, rng):
    base = rng.normal(0.35, 0.05, size=(size, size))
    k = 3
    pad = k // 2
    padded = np.pad(base, pad, mode="reflect")
    smoothed = np.zeros_like(base)
    for i in range(size):
        for j in range(size):
            smoothed[i, j] = padded[i:i + k, j:j + k].mean()
    return smoothed


def add_blob(img, cx, cy, radius, intensity, edge_seed_angle=0.0):
    size = img.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    angle = np.arctan2(yy - cy, xx - cx)
    jitter = 1 + 0.15 * np.sin(4 * angle + edge_seed_angle)
    mask = dist < (radius * jitter)
    out = img.copy()
    out[mask] += intensity
    return np.clip(out, 0, 1)


def make_dataset(n_per_class=60, size=IMG_SIZE, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_per_class):
        bg = tissue_background(size, rng)
        X.append(np.clip(bg + rng.normal(0, 0.03, size=(size, size)), 0, 1))
        y.append(0)
    for _ in range(n_per_class):
        bg = tissue_background(size, rng)
        cx = rng.integers(size // 4, 3 * size // 4)
        cy = rng.integers(size // 4, 3 * size // 4)
        radius = rng.uniform(3, 6)
        intensity = rng.uniform(0.35, 0.55)
        withtumor = add_blob(bg, cx, cy, radius, intensity, rng.uniform(0, 6.28))
        X.append(np.clip(withtumor + rng.normal(0, 0.03, size=(size, size)), 0, 1))
        y.append(1)
    X, y = np.array(X), np.array(y, dtype=float)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_custom_scan(has_tumor, blob_x_frac, blob_y_frac, blob_radius,
                      blob_intensity, noise_level, seed=0, size=IMG_SIZE):
    """Used by the interactive 'Try it yourself' tab: build one scan from slider values."""
    rng = np.random.default_rng(seed)
    bg = tissue_background(size, rng)
    if has_tumor:
        cx, cy = blob_x_frac * size, blob_y_frac * size
        bg = add_blob(bg, cx, cy, blob_radius, blob_intensity, edge_seed_angle=1.7)
    img = np.clip(bg + rng.normal(0, noise_level, size=(size, size)), 0, 1)
    return img


# ---------------------------------------------------------------------------
# CNN backbone (from-scratch NumPy, manual backprop) with a 4-unit bottleneck
# ---------------------------------------------------------------------------

class Conv2D:
    def __init__(self, num_filters, ksize=3, seed=0):
        rng = np.random.default_rng(seed)
        self.num_filters, self.ksize = num_filters, ksize
        self.filters = rng.normal(0, np.sqrt(2.0 / (ksize * ksize)),
                                   size=(num_filters, ksize, ksize))
        self.bias = np.zeros(num_filters)

    def _pad(self, x):
        p = self.ksize // 2
        return np.pad(x, ((p, p), (p, p)), mode="constant")

    def forward(self, x):
        self.x = x
        H, W = x.shape
        xp = self._pad(x)
        out = np.zeros((self.num_filters, H, W))
        for f in range(self.num_filters):
            k = self.filters[f]
            for i in range(H):
                for j in range(W):
                    region = xp[i:i + self.ksize, j:j + self.ksize]
                    out[f, i, j] = np.sum(region * k) + self.bias[f]
        return out

    def backward(self, d_out, lr):
        H, W = self.x.shape
        xp = self._pad(self.x)
        d_filters = np.zeros_like(self.filters)
        d_bias = np.zeros_like(self.bias)
        for f in range(self.num_filters):
            for i in range(H):
                for j in range(W):
                    region = xp[i:i + self.ksize, j:j + self.ksize]
                    d_filters[f] += d_out[f, i, j] * region
                    d_bias[f] += d_out[f, i, j]
        self.filters -= lr * d_filters
        self.bias -= lr * d_bias


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, d_out):
        return d_out * (1 - self.out ** 2)


class MaxPool2x2:
    def forward(self, x):
        self.x_shape = x.shape
        C, H, W = x.shape
        Ho, Wo = H // 2, W // 2
        out = np.zeros((C, Ho, Wo))
        self.argmax = np.zeros((C, Ho, Wo, 2), dtype=int)
        for c in range(C):
            for i in range(Ho):
                for j in range(Wo):
                    block = x[c, 2 * i:2 * i + 2, 2 * j:2 * j + 2]
                    idx = np.unravel_index(np.argmax(block), block.shape)
                    self.argmax[c, i, j] = idx
                    out[c, i, j] = block[idx]
        return out

    def backward(self, d_out):
        d_x = np.zeros(self.x_shape)
        Ho, Wo = d_out.shape[1], d_out.shape[2]
        for c in range(d_out.shape[0]):
            for i in range(Ho):
                for j in range(Wo):
                    di, dj = self.argmax[c, i, j]
                    d_x[c, 2 * i + di, 2 * j + dj] = d_out[c, i, j]
        return d_x


class Dense:
    def __init__(self, in_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d_out, lr):
        d_W = np.outer(self.x, d_out)
        d_x = self.W @ d_out
        self.W -= lr * d_W
        self.b -= lr * d_out
        return d_x


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class HybridBackbone:
    """Conv -> ReLU -> MaxPool -> Dense -> ReLU -> Dense(4) -> Tanh -> Dense(1) -> Sigmoid.
    The Dense(4)->Tanh bottleneck is what gets handed to the quantum circuit."""

    def __init__(self, img_size=IMG_SIZE, num_filters=8, hidden=16, bottleneck=4, seed=0):
        self.conv = Conv2D(num_filters, ksize=3, seed=seed)
        self.relu1 = ReLU()
        self.pool = MaxPool2x2()
        flat_dim = num_filters * (img_size // 2) * (img_size // 2)
        self.fc1 = Dense(flat_dim, hidden, seed=seed + 1)
        self.relu2 = ReLU()
        self.fc_bottleneck = Dense(hidden, bottleneck, seed=seed + 2)
        self.tanh = Tanh()
        self.fc_head = Dense(bottleneck, 1, seed=seed + 3)

    def forward(self, x, return_bottleneck=False):
        a = self.conv.forward(x)
        a = self.relu1.forward(a)
        a = self.pool.forward(a)
        self.pool_shape = a.shape
        flat = a.reshape(-1)
        a = self.fc1.forward(flat)
        a = self.relu2.forward(a)
        b = self.fc_bottleneck.forward(a)
        b = self.tanh.forward(b)
        if return_bottleneck:
            return b
        z = self.fc_head.forward(b)
        return sigmoid(z)[0]

    def backward(self, y_true, y_pred, lr):
        d_z = np.array([y_pred - y_true])
        d_b = self.fc_head.backward(d_z, lr)
        d_b = self.tanh.backward(d_b)
        d_a = self.fc_bottleneck.backward(d_b, lr)
        d_a = self.relu2.backward(d_a)
        d_flat = self.fc1.backward(d_a, lr)
        d_pool = d_flat.reshape(self.pool_shape)
        d_conv_out = self.pool.backward(d_pool)
        d_conv_out = self.relu1.backward(d_conv_out)
        self.conv.backward(d_conv_out, lr)

    def train_step(self, x, y, lr):
        p = self.forward(x)
        eps = 1e-9
        loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        self.backward(y, p, lr)
        return loss, p

    def embed(self, x):
        return self.forward(x, return_bottleneck=True)


# ---------------------------------------------------------------------------
# Quantum classifier head (PennyLane)
# ---------------------------------------------------------------------------

_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_dev)
def quantum_circuit(weights, x):
    qml.templates.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.templates.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def quantum_classifier(weights, bias, x):
    return quantum_circuit(weights, x) + bias


def to_angles(bottleneck_vec):
    """tanh output in [-1,1] -> rotation angles in [0, pi]."""
    return (bottleneck_vec + 1) / 2 * np.pi


# ---------------------------------------------------------------------------
# Full training pipeline (called once by the app, then cached)
# ---------------------------------------------------------------------------

def train_pipeline(n_per_class=60, seed=42, cnn_epochs=8, cnn_lr=0.05,
                    q_iters=40, q_batch=10, q_stepsize=0.3,
                    progress_cb=None):
    """Runs the whole two-stage pipeline and returns everything the UI needs.
    progress_cb(fraction, message) is called periodically if provided."""

    def report(frac, msg):
        if progress_cb is not None:
            progress_cb(frac, msg)

    np.random.seed(seed)
    X_img, y = make_dataset(n_per_class=n_per_class, size=IMG_SIZE, seed=seed)
    n_train = int(0.8 * len(X_img))
    X_train_img, y_train = X_img[:n_train], y[:n_train]
    X_test_img, y_test = X_img[n_train:], y[n_train:]

    # ---- Stage 1: CNN backbone ----
    backbone = HybridBackbone(img_size=IMG_SIZE, num_filters=8, hidden=16,
                               bottleneck=N_QUBITS, seed=seed)
    backbone_losses, backbone_accs = [], []
    for epoch in range(1, cnn_epochs + 1):
        perm = np.random.permutation(len(X_train_img))
        epoch_loss = 0.0
        for idx in perm:
            loss, _ = backbone.train_step(X_train_img[idx], y_train[idx], lr=cnn_lr)
            epoch_loss += loss
        epoch_loss /= len(X_train_img)
        correct = sum(int((backbone.forward(X_test_img[i]) > 0.5) == y_test[i])
                      for i in range(len(X_test_img)))
        acc = correct / len(X_test_img)
        backbone_losses.append(epoch_loss)
        backbone_accs.append(acc)
        report(0.1 + 0.55 * epoch / cnn_epochs,
               f"Training CNN backbone — epoch {epoch}/{cnn_epochs} (acc {acc:.0%})")

    # ---- Freeze, extract embeddings ----
    embed_train = np.array([backbone.embed(x) for x in X_train_img])
    embed_test = np.array([backbone.embed(x) for x in X_test_img])
    embed_train_q = to_angles(embed_train)
    embed_test_q = to_angles(embed_test)
    report(0.68, "Extracting learned features from the frozen CNN...")

    # ---- Stage 2: quantum head ----
    labels_train = 2 * y_train - 1
    labels_test = 2 * y_test - 1

    def cost(weights, bias, X, Y):
        loss = 0.0
        for i in range(len(X)):
            pred = quantum_classifier(weights, bias, X[i])
            # Use 'loss = loss + ...' rather than '+=' to ensure the graph 
            # tracks the operation properly in older Autograd versions
            loss = loss + (Y[i] - pred) ** 2
            
        return loss / len(X)
    def accuracy(Y, preds):
        s = [1 if p > 0 else -1 for p in preds]
        return float(np.mean(np.array(s) == np.array(Y)))

    pnp.random.seed(seed)
    q_weights = pnp.random.uniform(0, 2 * np.pi, size=(N_QLAYERS, N_QUBITS), requires_grad=True)
    q_bias = pnp.array(0.0, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=q_stepsize)

    q_history = {"iter": [], "cost": [], "test_acc": []}
    rng_np = np.random.default_rng(seed + 1)
    for it in range(1, q_iters + 1):
        bidx = rng_np.integers(0, len(embed_train_q), q_batch)
        q_weights, q_bias, _, _ = opt.step(cost, q_weights, q_bias,
                                            embed_train_q[bidx], labels_train[bidx])
        if it % max(1, q_iters // 8) == 0 or it == 1 or it == q_iters:
            preds_test = [quantum_classifier(q_weights, q_bias, x) for x in embed_test_q]
            c = float(cost(q_weights, q_bias, embed_train_q, labels_train))
            acc = accuracy(labels_test, preds_test)
            q_history["iter"].append(it)
            q_history["cost"].append(c)
            q_history["test_acc"].append(acc)
            report(0.7 + 0.28 * it / q_iters,
                   f"Training quantum head — iteration {it}/{q_iters} (acc {acc:.0%})")

    # ---- Final evaluation ----
    preds_test_raw = [quantum_classifier(q_weights, q_bias, x) for x in embed_test_q]
    preds_test = [1 if p > 0 else -1 for p in preds_test_raw]
    tp = fp = tn = fn = 0
    for p, t in zip(preds_test, labels_test):
        if p == 1 and t == 1: tp += 1
        elif p == 1 and t == -1: fp += 1
        elif p == -1 and t == -1: tn += 1
        else: fn += 1

    report(1.0, "Done.")

    return {
        "backbone": backbone,
        "q_weights": q_weights,
        "q_bias": q_bias,
        "X_img": X_img, "y": y,
        "X_train_img": X_train_img, "y_train": y_train,
        "X_test_img": X_test_img, "y_test": y_test,
        "embed_train": embed_train, "embed_test": embed_test,
        "backbone_losses": backbone_losses, "backbone_accs": backbone_accs,
        "q_history": q_history,
        "preds_test": preds_test,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def predict_single(backbone, q_weights, q_bias, image):
    """Run one image through the full frozen-CNN + quantum-head pipeline."""
    bottleneck = backbone.embed(image)
    angles = to_angles(bottleneck)
    raw = float(quantum_classifier(q_weights, q_bias, angles))
    label = "tumor" if raw > 0 else "normal"
    # how far the raw score is from the 0 decision boundary, normalized into [50%, 100%]
    confidence = float(np.clip(0.5 + abs(raw) / 2, 0.5, 1.0))
    return {
        "bottleneck": bottleneck,
        "raw_score": raw,
        "label": label,
        "confidence": confidence,
    }