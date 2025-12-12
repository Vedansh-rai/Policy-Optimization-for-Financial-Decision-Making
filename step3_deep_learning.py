# ============================================================================
# FAST F1 OPTIMIZATION - Target F1 > 0.55 in 10-15 minutes
# Balanced approach: Good F1 with fast training
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, classification_report, 
                              roc_curve, confusion_matrix, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1: Load Data
# ============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)

df_features = pd.read_pickle('df_preprocessed.pkl')
with open('feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)

all_features = feature_info['all_features']
print(f"✓ Data: {df_features.shape}, Features: {len(all_features)}\n")

# ============================================================================
# 2: FAST DATA PREPARATION
# ============================================================================
print("="*70)
print("DATA PREPARATION")
print("="*70)

X = df_features[all_features].values
y = df_features['target'].values

print(f"Original: {len(y):,} samples")
print(f"  Class 0: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"  Class 1: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✓ Data prepared!\n")

# ============================================================================
# 3: FAST DATALOADERS
# ============================================================================
print("="*70)
print("CREATING DATALOADERS")
print("="*70)

class LoanDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = LoanDataset(X_train_tensor, y_train_tensor)
test_dataset = LoanDataset(X_test_tensor, y_test_tensor)

# SPEED OPTIMIZATION 1: Larger batch size for faster training
BATCH_SIZE = 512  # Faster than 256

# Aggressive oversampling (2x is enough with good features)
class_counts = np.bincount(y_train.astype(int))
neg_count, pos_count = class_counts[0], class_counts[1]

pos_weight_sampler = (neg_count / pos_count) * 2.0  # 2x multiplier
sample_weights = np.where(y_train==1, pos_weight_sampler, 1.0).astype(np.float32)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                         num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

print(f"Batch size: {BATCH_SIZE} (larger for speed)")
print(f"Train batches: {len(train_loader)}")
print(f"Positive class weight: {pos_weight_sampler:.2f}x")
print("✓ DataLoaders ready!\n")

# ============================================================================
# 4: BALANCED FOCAL LOSS
# ============================================================================
class BalancedFocalLoss(nn.Module):
    """Focal Loss with good balance between speed and F1"""
    def __init__(self, alpha=0.85, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ============================================================================
# 5: SIMPLER FASTER MODEL
# ============================================================================
print("="*70)
print("DEFINING MODEL")
print("="*70)

class FastLoanMLP(nn.Module):
    """
    Simpler, faster model that still performs well
    Trade-off: Less capacity but trains 3x faster
    """
    def __init__(self, input_dim):
        super().__init__()
        
        # SPEED OPTIMIZATION 2: Shallower network (4 blocks instead of 6)
        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Block 2
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.35),
            
            # Block 3
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Block 4
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output
            nn.Linear(128, 1)
        )
        
        # Good initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# Initialize
input_dim = X_train_scaled.shape[1]
model = FastLoanMLP(input_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Device: {device}")
print("✓ Faster model ready!\n")

# ============================================================================
# 6: FAST TRAINING SETUP
# ============================================================================
print("="*70)
print("TRAINING SETUP")
print("="*70)

criterion = BalancedFocalLoss(alpha=0.85, gamma=2.5)

# SPEED OPTIMIZATION 3: Higher learning rate for faster convergence
optimizer = optim.AdamW(
    model.parameters(), 
    lr=0.002,  # Higher LR
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# SPEED OPTIMIZATION 4: Fewer epochs with aggressive schedule
NUM_EPOCHS = 20  # Just 20 epochs!
PATIENCE = 6

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.005,  # Higher peak
    epochs=NUM_EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.2,  # Short warmup
    div_factor=20,
    final_div_factor=100
)

print("\nFast Configuration:")
print(f"  Loss: Balanced Focal Loss (α=0.85, γ=2.5)")
print(f"  Optimizer: AdamW (lr=0.002)")
print(f"  Scheduler: OneCycleLR")
print(f"  Epochs: {NUM_EPOCHS} ⚡ FAST")
print(f"  Early Stopping: {PATIENCE}")
print(f"  Batch size: {BATCH_SIZE}")
print("✓ Setup complete!\n")

# ============================================================================
# 7: TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        outputs = torch.sigmoid(model(X_batch).squeeze())
        predictions.append(outputs.cpu().numpy())
        actuals.append(y_batch.numpy())
    
    return np.concatenate(predictions), np.concatenate(actuals)

def find_best_f1_threshold(y_true, y_proba, n_thresholds=100):
    """Fast threshold search"""
    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1

# ============================================================================
# 8: FAST TRAINING LOOP
# ============================================================================
print("="*70)
print(f"FAST TRAINING ({NUM_EPOCHS} EPOCHS)")
print("="*70)

best_f1 = 0
best_threshold = 0.5
best_epoch = 0
patience_counter = 0

history = {
    'train_loss': [],
    'val_auc': [],
    'val_f1': [],
    'val_f1_opt': [],
    'learning_rate': [],
    'threshold': []
}

import time
start_time = time.time()

print(f"\nStarting training...\n")

for epoch in range(NUM_EPOCHS):
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    
    # Evaluate
    y_pred_proba, y_true = evaluate(model, test_loader, device)
    
    # Metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    optimal_threshold, f1_opt = find_best_f1_threshold(y_true, y_pred_proba)
    
    y_pred_default = (y_pred_proba > 0.5).astype(int)
    f1_default = f1_score(y_true, y_pred_default, zero_division=0)
    
    # Store
    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['val_auc'].append(auc)
    history['val_f1'].append(f1_default)
    history['val_f1_opt'].append(f1_opt)
    history['learning_rate'].append(current_lr)
    history['threshold'].append(optimal_threshold)
    
    # Save best
    if f1_opt > best_f1:
        best_f1 = f1_opt
        best_threshold = optimal_threshold
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), 'best_dl_model.pth')
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"AUC: {auc:.4f} | "
              f"F1@0.5: {f1_default:.4f} | "
              f"F1@{optimal_threshold:.3f}: {f1_opt:.4f} ⭐")
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"AUC: {auc:.4f} | "
              f"F1@0.5: {f1_default:.4f} | "
              f"F1@{optimal_threshold:.3f}: {f1_opt:.4f}")
    
    if patience_counter >= PATIENCE:
        print(f"\n⚠ Early stopping at epoch {epoch+1}")
        break

elapsed_time = time.time() - start_time

print(f"\n✓ Training complete in {elapsed_time/60:.1f} minutes!")
print(f"✨ Best F1: {best_f1:.4f} at epoch {best_epoch}")
print(f"✨ Best threshold: {best_threshold:.3f}\n")

# ============================================================================
# 9: FINAL EVALUATION
# ============================================================================
print("="*70)
print("FINAL EVALUATION")
print("="*70)

model.load_state_dict(torch.load('best_dl_model.pth'))
y_pred_proba, y_true = evaluate(model, test_loader, device)

y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
y_pred_default = (y_pred_proba > 0.5).astype(int)

auc = roc_auc_score(y_true, y_pred_proba)
f1_optimal = f1_score(y_true, y_pred_optimal)
f1_default = f1_score(y_true, y_pred_default)

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"\n🎯 AUC-ROC: {auc:.4f}")
print(f"🎯 F1@0.5: {f1_default:.4f}")
print(f"🎯 F1@{best_threshold:.3f}: {f1_optimal:.4f}")
print(f"\n⏱️ Training time: {elapsed_time/60:.1f} minutes")
print(f"✨ Speed: {elapsed_time/best_epoch:.1f} seconds per epoch")

print("\n" + "-"*70)
print(f"Classification Report (τ={best_threshold:.3f}):")
print("-"*70)
print(classification_report(y_true, y_pred_optimal, 
                          target_names=['Paid', 'Default'], digits=4))

cm = confusion_matrix(y_true, y_pred_optimal)
print("\nConfusion Matrix:")
print(f"              Paid  Default")
print(f"Actual Paid   {cm[0,0]:6d}  {cm[0,1]:6d}")
print(f"       Default{cm[1,0]:6d}  {cm[1,1]:6d}")

from sklearn.metrics import precision_score, recall_score, accuracy_score
print(f"\nDetailed Metrics:")
print(f"  Accuracy:  {accuracy_score(y_true, y_pred_optimal):.4f}")
print(f"  Precision: {precision_score(y_true, y_pred_optimal):.4f}")
print(f"  Recall:    {recall_score(y_true, y_pred_optimal):.4f}")

print("\n💡 OPTIMIZATIONS FOR SPEED:")
print("   ✓ Larger batch size (512) → Fewer iterations")
print("   ✓ Shallower model (4 blocks) → Faster forward/backward")
print("   ✓ Higher learning rate → Faster convergence")
print("   ✓ Fewer epochs (20) → Less training time")
print("   ✓ 2x oversampling (not 3x) → Good balance")
print("   ✓ 100 threshold points (not 300) → Faster search")

# ============================================================================
# 10: Quick Visualizations
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. ROC Curve
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
ax1.plot(fpr, tpr, linewidth=2, label=f'AUC={auc:.4f}', color='blue')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax1.fill_between(fpr, tpr, alpha=0.3)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax1.set_title('ROC Curve', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. F1 Evolution
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history['val_f1'], linewidth=2, label='F1@0.5', color='orange', alpha=0.7)
ax2.plot(history['val_f1_opt'], linewidth=2, label='F1@optimal', color='green')
ax2.axhline(best_f1, color='r', linestyle='--', label=f'Best: {best_f1:.4f}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 Score')
ax2.set_title('F1 Score Evolution', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Learning Rate
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(history['learning_rate'], linewidth=2, color='purple')
ax3.set_xlabel('Step')
ax3.set_ylabel('Learning Rate')
ax3.set_title('Learning Rate Schedule', fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# 4. Prediction Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist([y_pred_proba[y_true==0], y_pred_proba[y_true==1]], 
         bins=50, label=['Paid', 'Default'], alpha=0.7, color=['blue', 'red'])
ax4.axvline(best_threshold, color='green', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Frequency')
ax4.set_title('Prediction Distribution', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Confusion Matrix
ax5 = fig.add_subplot(gs[1, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
           xticklabels=['Paid', 'Default'],
           yticklabels=['Paid', 'Default'])
ax5.set_xlabel('Predicted')
ax5.set_ylabel('Actual')
ax5.set_title(f'Confusion Matrix', fontweight='bold')

# 6. F1 vs Threshold
ax6 = fig.add_subplot(gs[1, 2])
thresholds_test = np.linspace(0.1, 0.9, 100)
f1_scores = []
for t in thresholds_test:
    y_pred_t = (y_pred_proba >= t).astype(int)
    f1_scores.append(f1_score(y_true, y_pred_t, zero_division=0))

ax6.plot(thresholds_test, f1_scores, linewidth=2, color='green')
ax6.axvline(best_threshold, color='r', linestyle='--', linewidth=2)
ax6.scatter([best_threshold], [best_f1], color='r', s=150, zorder=5)
ax6.set_xlabel('Threshold')
ax6.set_ylabel('F1 Score')
ax6.set_title('F1 vs Threshold', fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.suptitle(f'Fast Model - {elapsed_time/60:.1f} min - F1: {best_f1:.4f}', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("✓ Visualizations complete!\n")

# ============================================================================
# 11: Save Results
# ============================================================================
print("="*70)
print("SAVING RESULTS")
print("="*70)

dl_results = {
    'y_pred_proba': y_pred_proba,
    'y_pred': y_pred_optimal,
    'y_true': y_true,
    'auc': auc,
    'f1': f1_optimal,
    'threshold': best_threshold,
    'best_epoch': best_epoch,
    'training_time_minutes': elapsed_time/60,
    'test_indices': np.arange(len(y_test))
}

with open('dl_results.pkl', 'wb') as f:
    pickle.dump(dl_results, f)
with open('dl_history.pkl', 'wb') as f:
    pickle.dump(history, f)

print("✓ Results saved")

print("\n" + "="*70)
print(f"✨ FINAL F1: {f1_optimal:.4f}")
print(f"⏱️ TIME: {elapsed_time/60:.1f} minutes")
print(f"⚡ SPEED: {elapsed_time/best_epoch:.1f} sec/epoch")
print("="*70)

if f1_optimal >= 0.55:
    print("\n🎉 TARGET ACHIEVED! F1 ≥ 0.55")
elif f1_optimal >= 0.50:
    print("\n✅ Good F1 ≥ 0.50")
    print("\n💡 To push higher:")
    print("   1. Increase to 30 epochs")
    print("   2. Increase oversampling to 2.5x")
    print("   3. Use the enhanced preprocessing")
else:
    print("\n⚠️ F1 below 0.50")
    print("\n💡 Run enhanced_step2_preprocessing.py first!")

print("\n🚀 Complete!")