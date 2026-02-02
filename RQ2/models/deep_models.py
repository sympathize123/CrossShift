"""
Deep Learning Models for Tabular Data

These models use GPU acceleration and are NOT used with parallel computing.
All models are configured with similar parameter counts (~100K) for fair comparison.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neural_network import MLPClassifier

# Fix PyTorch 2.6+ weights_only issue
_original_torch_load = torch.load
def patched_torch_load(f, map_location=None, **kwargs):
    return _original_torch_load(f, map_location=map_location, weights_only=False, **kwargs)
torch.load = patched_torch_load

# Optional imports - will be checked at runtime
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from torch.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False

try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import NodeConfig
    from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
    PYTORCH_TABULAR_AVAILABLE = True
except ImportError:
    PYTORCH_TABULAR_AVAILABLE = False

try:
    from pytorch_widedeep.preprocessing import TabPreprocessor
    from pytorch_widedeep.models import TabResnet, WideDeep
    from pytorch_widedeep.training import Trainer
    from pytorch_widedeep.callbacks import EarlyStopping
    PYTORCH_WIDEDEEP_AVAILABLE = True
except ImportError:
    PYTORCH_WIDEDEEP_AVAILABLE = False


def ensure_float32_array(X):
    """Convert DataFrame/array to float32 numpy array"""
    if isinstance(X, pd.DataFrame):
        X = X.apply(pd.to_numeric, errors="coerce")
        arr = X.values
    else:
        arr = np.asarray(X)
        if arr.dtype == object:
            arr = pd.DataFrame(arr).apply(pd.to_numeric, errors="coerce").values
    
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def _to_loader(X, y=None, batch_size=256, shuffle=False):
    """Create simple PyTorch DataLoader"""
    X_t = torch.as_tensor(ensure_float32_array(X), dtype=torch.float32)
    if y is None:
        ds = TensorDataset(X_t)
    else:
        y_t = torch.as_tensor(y, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t)
    
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _perm_importance_binary(estimator, X, y, n_repeats=5, random_state=42):
    """Calculate permutation importance for binary classification"""
    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    base = roc_auc_score(y, estimator.predict_proba(X)[:, 1])
    imps = np.zeros(X.shape[1], dtype=float)
    
    for j in range(X.shape[1]):
        Xc = X.copy()
        scores = []
        for _ in range(n_repeats):
            rng.shuffle(Xc[:, j])
            p = estimator.predict_proba(Xc)[:, 1]
            try:
                scores.append(base - roc_auc_score(y, p))
            except ValueError:
                scores.append(0.0)
        imps[j] = np.mean(scores)
    
    imps = np.maximum(imps, 0)
    if imps.sum() == 0:
        imps[:] = 1.0
    return imps / imps.sum()


class MLPBackbone(nn.Module):
    """PyTorch MLP backbone"""
    def __init__(self, num_features, hidden_layer_sizes=(256, 128, 64, 32), 
                 activation='relu', dropout=0.1):
        super().__init__()
        layers = []
        prev_size = num_features
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def forward(self, x):
        return self.network(x).squeeze(-1)

    def get_embedding(self, x):
        """Evaluate only the backbone layers to get embeddings"""
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        # Process through all layers except the final linear head
        # The network is Sequential, so we need to iterate or slice
        out = x
        # self.network includes the final head. We need to stop before it.
        # Structure: Linear -> BN -> Act -> Dropout -> ... -> Linear(Head)
        # The last layer is the head (index -1)
        for layer in self.network[:-1]:
            out = layer(out)
        return out


# ===== Model with compile + AMP 안전 적용 =====
class MLPModel:
    """Multi-Layer Perceptron using PyTorch with torch.compile + AMP"""
    def __init__(self, num_features, device=None, epochs=300, batch_size=256,
                 lr=0.001, patience=100, hidden_layer_sizes=(256, 128, 64, 32),
                 activation='relu', dropout=0.0, alpha=0.001, random_state=42,
                 # NEW: 컴파일/AMP 관련 옵션
                 use_compile=False,
                 compile_mode='max-autotune',   # ['default', 'reduce-overhead', 'max-autotune', ...]
                 compile_dynamic=False,
                 prefer_bf16=True,              # Ampere+에서 bf16 우선
                 **kwargs):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # TF32 활성화 (Ampere+에서 matmul 가속)
        if self.device == 'cuda':
            torch.set_float32_matmul_precision('high')  # TF32 on
            torch.backends.cudnn.benchmark = True

        self.model = MLPBackbone(
            num_features=num_features,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            dropout=dropout
        ).to(self.device)

        # NEW: torch.compile 시도 (안되면 자동 우회)
        self._compiled = False
        if use_compile and hasattr(torch, "compile") and self.device == 'cuda':
            try:
                self.model = torch.compile(self.model, mode=compile_mode, dynamic=compile_dynamic)
                self._compiled = True
                print(f"[compile] enabled: mode={compile_mode}, dynamic={compile_dynamic}")
            except Exception as e:
                print(f"[compile] disabled (fallback): {e}")

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.alpha = alpha
        self.random_state = random_state
        self.best_state = None
        self.feature_importances_ = None
        self.num_features = num_features

        # NEW: AMP dtype/스케일러 결정
        self.use_amp = (self.device == 'cuda')
        self.amp_dtype = None
        self.scaler = None
        if self.use_amp:
            major, minor = torch.cuda.get_device_capability()
            # Ampere(8.0)+ 에서 bf16 안정적
            if prefer_bf16 and (major >= 8):
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # bf16은 스케일러 불필요
                print("[AMP] using bfloat16 autocast (no scaler)")
            else:
                self.amp_dtype = torch.float16
                self.scaler = GradScaler(enabled=True)
                print("[AMP] using float16 autocast + GradScaler")
        else:
            print("[AMP] disabled (using CPU or non-CUDA device)")
            print("[TF32] enabled on CUDA only; ignored here")

    def fit(self, X, y, X_val=None, y_val=None):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.alpha)
        crit = nn.BCEWithLogitsLoss()

        train_loader = _to_loader(X, y, self.batch_size, shuffle=True)
        val_loader = _to_loader(X_val, y_val, self.batch_size) if X_val is not None else None

        best_auc, bad = -np.inf, 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                # 더 효율적인 zero_grad
                opt.zero_grad(set_to_none=True)

                if self.use_amp:
                    # bf16: scaler 없음 / fp16: scaler 사용
                    if self.scaler is None:
                        with autocast(device_type='cuda', dtype=self.amp_dtype):
                            loss = crit(self.model(xb), yb)
                        loss.backward()
                        opt.step()
                    else:
                        with autocast(device_type='cuda', dtype=self.amp_dtype):
                            loss = crit(self.model(xb), yb)
                        self.scaler.scale(loss).backward()
                        self.scaler.step(opt)
                        self.scaler.update()
                else:
                    loss = crit(self.model(xb), yb)
                    loss.backward()
                    opt.step()

            # ===== Validation =====
            if val_loader is not None:
                self.model.eval()
                preds, ys = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device, non_blocking=True)
                        if self.use_amp:
                            with autocast(device_type='cuda', dtype=self.amp_dtype):
                                pred = torch.sigmoid(self.model(xb))
                        else:
                            pred = torch.sigmoid(self.model(xb))
                        # Ensure float32 before moving to numpy to avoid bfloat16 incompatibility
                        preds.append(pred.float().cpu().numpy())
                        ys.append(yb.numpy())

                if len(preds) == 0:
                    continue

                p = np.concatenate(preds)
                y_true = np.concatenate(ys)
                try:
                    auc = roc_auc_score(y_true, p)
                except ValueError:
                    auc = 0.0

                if auc > best_auc:
                    best_auc, bad = auc, 0
                    # GPU 메모리 압박 줄이기 위해 CPU에 저장
                    self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

        # Best state 로드
        if self.best_state:
            # 모듈이 compile된 상태든 아니든 state_dict 호환
            self.model.load_state_dict(self.best_state, strict=True)
            del self.best_state
            self.best_state = None

        # No feature importance calculation in fit
        self.feature_importances_ = None
        return self

    def predict_proba(self, X):
        self.model.eval()
        loader = _to_loader(X, None, self.batch_size, shuffle=False)
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=self.amp_dtype):
                        out = torch.sigmoid(self.model(xb))
                else:
                    out = torch.sigmoid(self.model(xb))
                # Cast to float32 to avoid bfloat16 numpy issues
                probs.append(out.float().cpu().numpy())
        p = np.concatenate(probs)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def get_embedding(self, X):
        """Get latent representations from the model"""
        self.model.eval()
        loader = _to_loader(X, None, self.batch_size, shuffle=False)
        embeddings = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=self.amp_dtype):
                        emb = self.model.get_embedding(xb)
                else:
                    emb = self.model.get_embedding(xb)
                embeddings.append(emb.cpu().numpy())
        return np.concatenate(embeddings)
    
    def calc_feature_importance(self, X, y, max_samples=1000):
        """Calculate feature importance using SHAP or permutation importance"""
        try:
            if not SHAP_AVAILABLE:
                raise ImportError("SHAP not available")
            
            # Sample data if too large for performance
            X_sample = X[:max_samples] if len(X) > max_samples else X
            
            # Create SHAP explainer
            explainer = shap.Explainer(self.predict_proba, X_sample)
            shap_values = explainer(X_sample)
            
            # Use absolute mean SHAP values for feature importance
            if hasattr(shap_values, 'values'):
                shap_vals = shap_values.values[:, :, 1]  # Use positive class
            else:
                shap_vals = shap_values[:, :, 1]
            
            importance = np.abs(shap_vals).mean(axis=0)
            
            # Normalize to sum to 1
            self.feature_importances_ = importance / importance.sum()
            
        except Exception as e:
            print(f"SHAP failed ({e}), using permutation importance fallback")
            # Fallback to permutation importance
            try:
                self.feature_importances_ = _perm_importance_binary(
                    self, X, y, n_repeats=1, random_state=self.random_state
                )
            except Exception:
                # Final fallback to uniform importance
                self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        
        return self.feature_importances_


class TabTransformerBackbone(nn.Module):
    """Optimized TabTransformer architecture for tabular data"""
    
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=4, dropout=0.0):
        super().__init__()
        self.num_features = num_features
        # Project each feature to d_model dimension
        self.value_proj = nn.Linear(1, d_model)
        # Column embeddings for positional encoding
        self.col_embed = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)
        
        # Optimized Transformer encoder layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=2*d_model,  # Reduced from 4x to 2x for speed
            dropout=dropout,
            batch_first=True, 
            activation='gelu',
            norm_first=True  # Pre-norm for better convergence
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for faster convergence"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        
        B = x.size(0)
        # Create feature tokens: (B, F, 1) -> (B, F, d_model)
        tok = self.value_proj(x.unsqueeze(-1)) + self.col_embed
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        
        # Apply transformer encoder
        enc = self.encoder(seq)
        # Use CLS token for classification
        cls_out = self.norm(enc[:, 0])
        return self.head(cls_out).squeeze(-1)

    def get_embedding(self, x):
        """Get the CLS token embedding"""
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        
        B = x.size(0)
        tok = self.value_proj(x.unsqueeze(-1)) + self.col_embed
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        
        enc = self.encoder(seq)
        return self.norm(enc[:, 0])


class TabTransformerModel:
    """TabTransformer Model with GPU support"""
    
    def __init__(self, num_features, device=None, epochs=300, batch_size=256,
                 lr=1e-3, patience=100, d_model=64, nhead=8, num_layers=6, 
                 dropout=0.0, random_state=42, **kwargs):
        
        # Automatically detect and use GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(random_state)
        # Smaller model for faster training
        self.model = TabTransformerBackbone(
            num_features, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            dropout=dropout
        )
        self.model.to(self.device)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.best_state = None
        self.feature_importances_ = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        print(self.device)
        train_loader = _to_loader(X, y, self.batch_size, shuffle=True)
        val_loader = _to_loader(X_val, y_val, self.batch_size)
        
        best_auc, bad = -np.inf, 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                opt.step()
                
            # Fast validation
            self.model.eval()
            preds, ys = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    pred = torch.sigmoid(self.model(xb))
                    preds.append(pred.cpu().numpy())
                    ys.append(yb.numpy())
            
            if len(preds) == 0:
                continue
                
            p = np.concatenate(preds)
            y_true = np.concatenate(ys)
            
            try:
                auc = roc_auc_score(y_true, p)
            except ValueError:
                auc = 0.0
                
            if auc > best_auc:
                best_auc, bad = auc, 0
                # Store on CPU to free GPU memory for training
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        
        # Load best model with minimal overhead
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            del self.best_state  # Immediate cleanup
            self.best_state = None
            
        # Skip GPU cache cleanup - let PyTorch manage memory automatically
            
        # No feature importance calculation in fit - use calc_feature_importance() separately
        self.feature_importances_ = None
        
        return self
        
    def predict_proba(self, X):
        self.model.eval()
        loader = _to_loader(X, None, self.batch_size, shuffle=False)
        probs = []
        
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                probs.append(torch.sigmoid(self.model(xb)).cpu().numpy())
        
        p = np.concatenate(probs)
        return np.vstack([1 - p, p]).T
    
    def get_embedding(self, X):
        """Get latent representations from the model"""
        self.model.eval()
        loader = _to_loader(X, None, self.batch_size, shuffle=False)
        embeddings = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                embeddings.append(self.model.get_embedding(xb).cpu().numpy())
        return np.concatenate(embeddings)
    
    def calc_feature_importance(self, X, y, max_samples=1000):
        """Calculate feature importance using SHAP"""
        try:
            if not SHAP_AVAILABLE:
                raise ImportError("SHAP not available")
            
            # Sample data if too large for performance
            X_sample = X[:max_samples] if len(X) > max_samples else X
            
            # Create SHAP explainer
            explainer = shap.Explainer(self.predict_proba, X_sample)
            shap_values = explainer(X_sample)
            
            # Use absolute mean SHAP values for feature importance
            if hasattr(shap_values, 'values'):
                # For newer SHAP versions
                shap_vals = shap_values.values[:, :, 1]  # Use positive class
            else:
                shap_vals = shap_values[:, :, 1]
            
            importance = np.abs(shap_vals).mean(axis=0)
            
            # Normalize to sum to 1
            self.feature_importances_ = importance / importance.sum()
            
        except Exception as e:
            print(f"SHAP failed ({e}), using permutation importance fallback")
            # Fallback to permutation importance
            try:
                self.feature_importances_ = _perm_importance_binary(
                    self, X, y, n_repeats=1, random_state=self.random_state
                )
            except Exception:
                # Final fallback to uniform importance
                self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        
        return self.feature_importances_


class PreNormBlock(nn.Module):
    """Pre-normalization transformer block"""
    
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.do(self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0])
        x = x + self.do(self.ff(self.ln2(x)))
        return x


class OptimizedPreNormBlock(nn.Module):
    """Optimized pre-normalization transformer block for faster training"""
    
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, 
            dropout=dropout, 
            batch_first=True,
            bias=False  # Remove bias for speed
        )
        # Reduced FFN size from 4x to 2x for speed
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2*d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model, d_model, bias=False)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        x_norm = self.ln1(x)
        x = x + self.do(self.attn(x_norm, x_norm, x_norm)[0])
        x = x + self.do(self.ff(self.ln2(x)))
        return x


class FTTransformerBackbone(nn.Module):
    """Optimized Feature Tokenizer Transformer (FT-Transformer) architecture"""
    
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=4, dropout=0.0):
        super().__init__()
        self.num_features = num_features
        self.value_proj = nn.Linear(1, d_model)
        self.feature_embed = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)
        
        # Pre-norm transformer blocks with reduced FFN size
        self.blocks = nn.ModuleList([
            OptimizedPreNormBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for faster convergence"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            
        B = x.size(0)
        tok = self.value_proj(x.unsqueeze(-1)) + self.feature_embed
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        
        for blk in self.blocks:
            seq = blk(seq)
            
        cls_out = self.head_norm(seq[:, 0])
        return self.head(cls_out).squeeze(-1)

    def get_embedding(self, x):
        """Get the CLS token embedding"""
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            
        B = x.size(0)
        tok = self.value_proj(x.unsqueeze(-1)) + self.feature_embed
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        
        for blk in self.blocks:
            seq = blk(seq)
            
        return self.head_norm(seq[:, 0])


class FTTransformerModel:
    """FT-Transformer Model with GPU support"""
    
    def __init__(self, num_features, device=None, epochs=300, batch_size=128,
                 lr=1e-3, patience=100, d_model=128, nhead=8, num_layers=4, 
                 dropout=0.0, random_state=42, **kwargs):
        
        # Automatically detect and use GPU if available  
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"FT-Transformer using device: {self.device}")
        
        torch.manual_seed(random_state)
        # Smaller model for faster training
        self.model = FTTransformerBackbone(
            num_features,
            d_model=d_model,
            nhead=nhead, 
            num_layers=num_layers,
            dropout=dropout
        )
        self.model.to(self.device)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.best_state = None
        self.feature_importances_ = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        # Disable compilation by default for stability
        compile_enabled = False  # Default to disabled
        if compile_enabled:
            try:
                if hasattr(torch, 'compile') and self.device == 'cuda':
                    self.model = torch.compile(self.model, mode='default', dynamic=False)
                    print("FT-Transformer compiled for acceleration")
            except Exception as e:
                print(f"Model compilation failed, continuing without: {e}")
        else:
            print("FT-Transformer compilation disabled for stability")
        
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        
        # Conservative mixed precision - disable by default for stability
        use_amp = True  # self.device == 'cuda' and enable_amp
        scaler = None
        if use_amp:
            if AMP_AVAILABLE:
                try:
                    scaler = GradScaler('cuda')
                    print("FT-Transformer using mixed precision training")
                except Exception as e:
                    try:
                        scaler = GradScaler()
                        print("FT-Transformer using mixed precision training")
                    except Exception as e:
                        use_amp = False
                        print(f"Mixed precision disabled: {e}")
            else:
                use_amp = False
                print("Mixed precision disabled: torch.amp not available")
        else:
            print("FT-Transformer using standard precision")
        
        train_loader = _to_loader(X, y, self.batch_size, shuffle=True)
        val_loader = _to_loader(X_val, y_val, self.batch_size)
        
        best_auc, bad = -np.inf, 0
        
        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                
                if use_amp:
                    with autocast('cuda' if self.device == 'cuda' else 'cpu'):
                        loss = crit(self.model(xb), yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                else:
                    opt.zero_grad()
                    loss = crit(self.model(xb), yb)
                    loss.backward()
                    opt.step()
            
            self.model.eval()
            preds, ys = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    if use_amp:
                        with autocast('cuda' if self.device == 'cuda' else 'cpu'):
                            pred = torch.sigmoid(self.model(xb))
                    else:
                        pred = torch.sigmoid(self.model(xb))
                    preds.append(pred.cpu().numpy())
                    ys.append(yb.numpy())
            
            p = np.concatenate(preds)
            y_true = np.concatenate(ys)
            
            try:
                auc = roc_auc_score(y_true, p)
            except ValueError:
                auc = 0.0
                
            if auc > best_auc:
                best_auc, bad = auc, 0
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        
        # Load best model
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            
        # No feature importance calculation in fit - use calc_feature_importance() separately
        self.feature_importances_ = None
        return self
        
    def predict_proba(self, X):
        self.model.eval()
        loader = _to_loader(X, None, self.batch_size, shuffle=False)
        probs = []
        
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                probs.append(torch.sigmoid(self.model(xb)).cpu().numpy())
        
        p = np.concatenate(probs)
        return np.vstack([1 - p, p]).T
    
    def get_embedding(self, X):
        """Get latent representations from the model"""
        self.model.eval()
        loader = _to_loader(X, None, self.batch_size, shuffle=False)
        embeddings = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                embeddings.append(self.model.get_embedding(xb).cpu().numpy())
        return np.concatenate(embeddings)
    
    def calc_feature_importance(self, X, y, max_samples=1000):
        """Calculate feature importance using SHAP"""
        try:
            if not SHAP_AVAILABLE:
                raise ImportError("SHAP not available")
            
            # Sample data if too large for performance
            X_sample = X[:max_samples] if len(X) > max_samples else X
            
            # Create SHAP explainer
            explainer = shap.Explainer(self.predict_proba, X_sample)
            shap_values = explainer(X_sample)
            
            # Use absolute mean SHAP values for feature importance
            if hasattr(shap_values, 'values'):
                shap_vals = shap_values.values[:, :, 1]  # Use positive class
            else:
                shap_vals = shap_values[:, :, 1]
            
            importance = np.abs(shap_vals).mean(axis=0)
            
            # Normalize to sum to 1
            self.feature_importances_ = importance / importance.sum()
            
        except Exception as e:
            print(f"SHAP failed ({e}), using permutation importance fallback")
            # Fallback to permutation importance
            try:
                self.feature_importances_ = _perm_importance_binary(
                    self, X, y, n_repeats=1, random_state=self.random_state
                )
            except Exception:
                # Final fallback to uniform importance
                self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        
        return self.feature_importances_


class NODEModel:
    """NODE (Neural Oblivious Decision Ensembles) using pytorch-tabular"""
    
    def __init__(self, num_features, device=None, random_state=42, 
                 num_layers=2, num_trees=128, depth=4, batch_size=128, 
                 max_epochs=200, early_stopping_patience=50, learning_rate=1e-3, **kwargs):
        if not PYTORCH_TABULAR_AVAILABLE:
            raise ImportError("pytorch-tabular is required for NODE model")
        
        # Automatically detect and use GPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NODE using device: {self.device}")
        
        # Smaller, user-configurable parameters
        self.data_cfg = DataConfig(
            target=["label"],
            continuous_cols=[f"f{i}" for i in range(num_features)],
            categorical_cols=[],
            validation_split=0.0,
        )
        
        self.model_cfg = NodeConfig(
            task="classification",
            num_layers=num_layers,
            num_trees=num_trees,
            depth=depth,
            input_dropout=kwargs.get('input_dropout', 0.1),
            learning_rate=learning_rate
        )
        
        self.trainer_cfg = TrainerConfig(
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=kwargs.get('early_stopping_min_delta', 1e-4),
            check_val_every_n_epoch=1,
            seed=random_state,
            accelerator="gpu" if self.device == "cuda" else "cpu"
        )
        
        self.opt_cfg = OptimizerConfig()
        self.num_features = num_features
        self.feature_importances_ = None
        
        self.model = TabularModel(
            data_config=self.data_cfg,
            model_config=self.model_cfg,
            optimizer_config=self.opt_cfg,
            trainer_config=self.trainer_cfg,
            experiment_config=None,
            suppress_lightning_logger=True,
            verbose=False
        )
        
    def fit(self, X, y, X_val=None, y_val=None):
        def _to_df(X, y):
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["label"] = y.astype(int)
            return df
            
        train_df = _to_df(np.asarray(X), np.asarray(y))
        val_df = None if X_val is None else _to_df(np.asarray(X_val), np.asarray(y_val))
        
        self.model.fit(train=train_df, validation=val_df)
        
        # No feature importance calculation in fit
        self.feature_importances_ = None
        return self
        
    def predict_proba(self, X):
        df = pd.DataFrame(np.asarray(X), columns=[f"f{i}" for i in range(X.shape[1])])
        
        # Suppress pytorch-tabular deprecation warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", 
                                  message="Classification prediction column will be renamed*",
                                  category=DeprecationWarning)
            warnings.filterwarnings("ignore", 
                                  message="`include_input_features` will be deprecated*",
                                  category=DeprecationWarning)
            pred_df = self.model.predict(df)
        
        # Extract probabilities using the expected future column name first
        if "label_prediction" in pred_df.columns:
            p = pred_df["label_prediction"].to_numpy()
        elif "1_probability" in pred_df.columns:
            p = pred_df["1_probability"].to_numpy()
        elif "prediction_probability" in pred_df.columns:
            p = pred_df["prediction_probability"].to_numpy()
        else:
            proba_cols = [c for c in pred_df.columns if c.endswith("_probability")]
            if proba_cols:
                p = pred_df[proba_cols[-1]].to_numpy()
            else:
                p = pred_df.to_numpy().ravel().astype(float)
                
        return np.vstack([1 - p, p]).T
    
    def calc_feature_importance(self, X, y, max_samples=1000):
        """Calculate feature importance using SHAP or permutation importance"""
        try:
            if not SHAP_AVAILABLE:
                raise ImportError("SHAP not available")
            
            # Sample data if too large for performance
            X_sample = X[:max_samples] if len(X) > max_samples else X
            
            # Create SHAP explainer
            explainer = shap.Explainer(self.predict_proba, X_sample)
            shap_values = explainer(X_sample)
            
            # Use absolute mean SHAP values for feature importance
            if hasattr(shap_values, 'values'):
                shap_vals = shap_values.values[:, :, 1]  # Use positive class
            else:
                shap_vals = shap_values[:, :, 1]
            
            importance = np.abs(shap_vals).mean(axis=0)
            
            # Normalize to sum to 1
            self.feature_importances_ = importance / importance.sum()
            
        except Exception as e:
            print(f"SHAP failed ({e}), using permutation importance fallback")
            # Fallback to permutation importance
            try:
                self.feature_importances_ = _perm_importance_binary(
                    self, X, y, n_repeats=1, random_state=42
                )
            except Exception:
                # Final fallback to uniform importance
                self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        
        return self.feature_importances_


class TabResNetModel:
    """TabResNet using pytorch-widedeep"""
    
    def __init__(self, num_features, device=None, epochs=300, batch_size=128,
                 lr=1e-3, patience=100, blocks_dims=(128, 64), mlp_hidden_dims=(32, 16),
                 blocks_dropout=0.0, mlp_dropout=0.0, random_state=42, **kwargs):
        if not PYTORCH_WIDEDEEP_AVAILABLE:
            raise ImportError("pytorch-widedeep is required for TabResNet model")
        
        # Automatically detect and use GPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") 
        print(f"TabResNet using device: {self.device}")
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.num_features = num_features
        
        # Smaller, user-configurable parameters - ensure they're lists
        self.blocks_dims = list(blocks_dims)
        self.blocks_dropout = blocks_dropout
        self.mlp_hidden_dims = list(mlp_hidden_dims)
        self.mlp_dropout = mlp_dropout
        
        self._cont_cols = None
        self._preproc = None
        self._model = None
        self._trainer = None
        self.feature_importances_ = None
        
    def _ensure_float32_df(self, X):
        """Convert input to float32 DataFrame"""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(np.asarray(X), columns=[f"f{i}" for i in range(X.shape[1])])
        
        # Force numeric
        for c in df.columns:
            if not np.issubdtype(df[c].dtype, np.number):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.astype(np.float32)
        
    def _build(self, Xdf):
        """Build model architecture"""
        # Preprocessor - use new parameter name to avoid deprecation warning
        self._cont_cols = list(Xdf.columns)
        self._preproc = TabPreprocessor(
            continuous_cols=self._cont_cols,
            cols_to_scale=self._cont_cols,
            verbose=0
        )
        X_tab = self._preproc.fit_transform(Xdf)
        
        # TabResnet model
        tab_resnet = TabResnet(
            column_idx=self._preproc.column_idx,
            continuous_cols=self._cont_cols,
            blocks_dims=self.blocks_dims,
            blocks_dropout=self.blocks_dropout,
            mlp_hidden_dims=self.mlp_hidden_dims,
            mlp_dropout=self.mlp_dropout,
        )
        
        self._model = WideDeep(deeptabular=tab_resnet)
        
        # Trainer with GPU support - use train_loss for early stopping if no validation
        self._trainer = Trainer(
            model=self._model,
            objective="binary",
            metrics=None,
            verbose=0,
            seed=self.random_state,
            num_workers=0,
            callbacks=[EarlyStopping(patience=self.patience, monitor="train_loss")]
        )
        
    def fit(self, X, y, X_val=None, y_val=None):
        Xdf = self._ensure_float32_df(X)
        self._build(Xdf)
        
        Xv = self._ensure_float32_df(X_val)
        X_tab_val = self._preproc.transform(Xv)
        
        self._trainer.fit(
            X_tab=self._preproc.transform(Xdf),
            target=y.astype(np.float32),
            X_tab_val=X_tab_val,
            target_val=y_val.astype(np.float32),
            n_epochs=self.epochs,
            batch_size=self.batch_size,
            val_batch_size=self.batch_size
        )
        
        # No feature importance calculation in fit
        self.feature_importances_ = None
        return self
        
    def predict_proba(self, X):
        Xdf = self._ensure_float32_df(X)
        X_tab = self._preproc.transform(Xdf)
        preds = self._trainer.predict(X_tab=X_tab)
        
        # Handle different prediction formats
        if hasattr(preds, 'flatten'):
            p = preds.flatten()
        else:
            p = np.asarray(preds).flatten()
            
        # Ensure probabilities are in [0,1]
        p = np.clip(p, 0, 1)
        return np.vstack([1 - p, p]).T
    
    def calc_feature_importance(self, X, y, max_samples=1000):
        """Calculate feature importance using SHAP or permutation importance"""
        try:
            if not SHAP_AVAILABLE:
                raise ImportError("SHAP not available")
            
            # Sample data if too large for performance
            X_sample = X[:max_samples] if len(X) > max_samples else X
            
            # Create SHAP explainer
            explainer = shap.Explainer(self.predict_proba, X_sample)
            shap_values = explainer(X_sample)
            
            # Use absolute mean SHAP values for feature importance
            if hasattr(shap_values, 'values'):
                shap_vals = shap_values.values[:, :, 1]  # Use positive class
            else:
                shap_vals = shap_values[:, :, 1]
            
            importance = np.abs(shap_vals).mean(axis=0)
            
            # Normalize to sum to 1
            self.feature_importances_ = importance / importance.sum()
            
        except Exception as e:
            print(f"SHAP failed ({e}), using permutation importance fallback")
            # Fallback to permutation importance
            try:
                self.feature_importances_ = _perm_importance_binary(
                    self, X, y, n_repeats=1, random_state=self.random_state
                )
            except Exception:
                # Final fallback to uniform importance
                self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        
        return self.feature_importances_
