# ============================================================================
# FIXED OFFLINE REINFORCEMENT LEARNING
# Addresses: selection bias, reward signal, and policy learning
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import d3rlpy
from sklearn.model_selection import train_test_split
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================================
# 1: Load Data & DL Model Predictions
# ============================================================================
print("="*70)
print("LOADING DATA & DL PREDICTIONS")
print("="*70)

df_features = pd.read_pickle('df_preprocessed.pkl')
with open('feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load DL predictions to add synthetic deny examples
with open('dl_results.pkl', 'rb') as f:
    dl_results = pickle.load(f)

all_features = feature_info['all_features']
print(f"✓ Data: {df_features.shape}, Features: {len(all_features)}\n")

# ============================================================================
# 2: FIX #1 - Create Synthetic Deny Examples Using DL Model
# ============================================================================
print("="*70)
print("CREATING SYNTHETIC DENY EXAMPLES")
print("="*70)

df_rl = df_features.copy()

# Calculate actual rewards for approved loans
def calculate_reward(row):
    loan_amount = float(row['loan_amnt'].iloc[0] if isinstance(row['loan_amnt'], pd.Series) else row['loan_amnt'])
    interest_rate = float(row['int_rate'].iloc[0] if isinstance(row['int_rate'], pd.Series) else row['int_rate']) / 100
    outcome = float(row['target'].iloc[0] if isinstance(row['target'], pd.Series) else row['target'])
    
    if outcome == 0:  # Fully Paid
        # 3-year interest profit
        return loan_amount * interest_rate * 3
    else:  # Default
        # 20% recovery rate
        return -(loan_amount * 0.8)

df_rl['reward_approve'] = df_rl.apply(calculate_reward, axis=1)

# FIX: Create synthetic DL predictions for all samples
# Use DL predictions on test set, extrapolate for full dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Train a simple model on DL predictions to predict default for all data
X_all = df_rl[all_features].values
X_test = dl_results.get('X_test', X_all[-len(dl_results['y_pred_proba']):])
y_test = dl_results['y_true']
y_pred_proba_test = dl_results['y_pred_proba']

# Use logistic regression to estimate default probs for full dataset
lr = LogisticRegression(max_iter=1000)
try:
    lr.fit(X_test if len(X_test) == len(y_test) else X_all[-len(y_test):], y_test)
    df_rl['dl_default_prob'] = lr.predict_proba(X_all)[:, 1]
except:
    # Fallback: use simple heuristic based on features
    df_rl['dl_default_prob'] = 0.2  # Default to baseline 20% default rate

# Deny rewards: positive for avoiding bad loans, small negative for rejecting good ones
# For loans that would default (target=1): denying saves the loss
default_mask = df_rl['target'] == 1
df_rl.loc[default_mask, 'reward_deny'] = df_rl.loc[default_mask, 'loan_amnt'] * 0.8  # Avoided 80% loss

# For loans that would be paid (target=0): denying loses the interest profit
paid_mask = df_rl['target'] == 0
df_rl.loc[paid_mask, 'reward_deny'] = -(df_rl.loc[paid_mask, 'loan_amnt'] * df_rl.loc[paid_mask, 'int_rate'] / 100 * 0.5)  # Lost half the interest

print(f"\n✓ Created deny rewards:")
print(f"  Loans that would default: {default_mask.sum():,} (denying = +reward)")
print(f"  Loans that would be paid: {paid_mask.sum():,} (denying = -reward)")
print(f"  Avg deny reward for defaults: ${df_rl.loc[default_mask, 'reward_deny'].mean():.2f}")
print(f"  Avg deny reward for paid: ${df_rl.loc[paid_mask, 'reward_deny'].mean():.2f}")

# ============================================================================
# 3: FIX #2 - Create Balanced Dataset with Both Actions
# ============================================================================
print("\n" + "="*70)
print("CREATING BALANCED ACTION DATASET")
print("="*70)

# Sample loans to create a balanced dataset
SAMPLE_SIZE = 100000  # Increased for more training data
df_sample = df_rl.sample(n=min(SAMPLE_SIZE, len(df_rl)), random_state=42)

# Create approve examples (actual historical data)
states_approve = df_sample[all_features].values
actions_approve = np.ones(len(states_approve))
rewards_approve = df_sample['reward_approve'].values

# Create deny examples (synthetic) - focus on loans where denying is beneficial
# Strategy 1: High-risk loans (likely to default)
high_risk_sample = df_sample[(df_sample['dl_default_prob'] > 0.5) | (df_sample['target'] == 1)]
# Strategy 2: Include some lower-risk to teach the agent trade-offs
medium_risk_candidates = df_sample[(df_sample['dl_default_prob'] > 0.3) & (df_sample['dl_default_prob'] <= 0.5)]
n_medium_sample = min(5000, len(df_sample)//20, len(medium_risk_candidates))
medium_risk_sample = medium_risk_candidates.sample(n=n_medium_sample, random_state=42) if n_medium_sample > 0 else pd.DataFrame()

combined_deny_candidates = pd.concat([high_risk_sample, medium_risk_sample]).drop_duplicates()
n_deny = min(len(combined_deny_candidates), len(states_approve) // 2)  # 1:2 ratio (more denies)

states_deny = combined_deny_candidates[all_features].values[:n_deny]
actions_deny = np.zeros(n_deny)
rewards_deny = combined_deny_candidates['reward_deny'].values[:n_deny]

# Combine
states = np.vstack([states_approve, states_deny])
actions = np.concatenate([actions_approve, actions_deny])
rewards = np.concatenate([rewards_approve, rewards_deny])
terminals = np.ones(len(states))

print(f"\n✓ Balanced dataset created:")
print(f"  Approve examples: {len(actions_approve):,}")
print(f"  Deny examples: {n_deny:,}")
print(f"  Total: {len(actions):,}")
print(f"  Deny ratio: {n_deny/len(actions)*100:.1f}%")

# ============================================================================
# 4: FIX #3 - Normalize Rewards Properly
# ============================================================================
print("\n" + "="*70)
print("NORMALIZING REWARDS")
print("="*70)

print(f"\nOriginal rewards:")
print(f"  Mean: ${rewards.mean():.2f}")
print(f"  Std: ${rewards.std():.2f}")
print(f"  Min: ${rewards.min():.2f}")
print(f"  Max: ${rewards.max():.2f}")

# Clip extreme values first
reward_p5 = np.percentile(rewards, 5)
reward_p95 = np.percentile(rewards, 95)
rewards_clipped = np.clip(rewards, reward_p5, reward_p95)

# Then normalize
reward_mean = rewards_clipped.mean()
reward_std = rewards_clipped.std()
rewards_normalized = (rewards_clipped - reward_mean) / (reward_std + 1e-8)

print(f"\nNormalized rewards:")
print(f"  Mean: {rewards_normalized.mean():.2f}")
print(f"  Std: {rewards_normalized.std():.2f}")
print(f"  Min: {rewards_normalized.min():.2f}")
print(f"  Max: {rewards_normalized.max():.2f}")

# ============================================================================
# 5: Prepare Dataset & Split
# ============================================================================
print("\n" + "="*70)
print("PREPARING DATASET")
print("="*70)

# Scale states
states_scaled = scaler.transform(states)

# Train-test split
train_idx, test_idx = train_test_split(
    np.arange(len(states_scaled)), 
    test_size=0.2, 
    random_state=42,
    stratify=actions  # Ensure balanced split
)

states_train = states_scaled[train_idx]
actions_train = actions[train_idx]
rewards_train = rewards_normalized[train_idx]
terminals_train = terminals[train_idx]

states_test = states_scaled[test_idx]
actions_test = actions[test_idx]
rewards_test = rewards[test_idx]  # Keep original for evaluation

print(f"\n✓ Train set: {len(states_train):,}")
print(f"  Approve: {(actions_train == 1).sum():,}")
print(f"  Deny: {(actions_train == 0).sum():,}")
print(f"\n✓ Test set: {len(states_test):,}")
print(f"  Approve: {(actions_test == 1).sum():,}")
print(f"  Deny: {(actions_test == 0).sum():,}")

# Create d3rlpy dataset
dataset = d3rlpy.dataset.MDPDataset(
    observations=states_train,
    actions=actions_train.reshape(-1, 1),
    rewards=rewards_train,
    terminals=terminals_train
)

# ============================================================================
# 6: FIX #4 - Train with Higher Alpha (More Conservative)
# ============================================================================
print("\n" + "="*70)
print("TRAINING CONSERVATIVE RL AGENT")
print("="*70)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device_str}")

# Higher alpha = more conservative = less likely to approve blindly
cql = d3rlpy.algos.DiscreteCQLConfig(
    batch_size=512,      # Larger batch for stability
    learning_rate=1e-4,  # Lower LR for stability
    n_critics=2,         # 2 critics for better Q-value estimation
    alpha=10.0           # VERY HIGH alpha for conservative policy
).create(device=device_str)

print(f"\nConfiguration:")
print(f"  Batch size: 512")
print(f"  Learning rate: 1e-4")
print(f"  Critics: 2")
print(f"  Alpha: 10.0 (EXTREMELY conservative)")

N_STEPS = 20000
STEPS_PER_EPOCH = 1000

print(f"\nTraining for {N_STEPS:,} steps...")
start_time = time.time()

cql.fit(
    dataset,
    n_steps=N_STEPS,
    n_steps_per_epoch=STEPS_PER_EPOCH,
    evaluators={
        'td_error': d3rlpy.metrics.TDErrorEvaluator(),
    },
    show_progress=True
)

training_time = time.time() - start_time
print(f"\n✓ Training complete in {training_time/60:.1f} minutes!")

# Save model
cql.save_model('cql_loan_agent_fixed.pt')
print("✓ Model saved\n")

# ============================================================================
# 7: Evaluate Fixed Policy
# ============================================================================
print("="*70)
print("EVALUATING FIXED POLICY")
print("="*70)

# Get policy actions
policy_actions = cql.predict(states_test).flatten()

deny_mask = policy_actions == 0
approve_mask = policy_actions == 1

print(f"\n📊 Policy Decisions:")
print(f"  Deny:    {deny_mask.sum():,} ({deny_mask.sum()/len(policy_actions)*100:.1f}%)")
print(f"  Approve: {approve_mask.sum():,} ({approve_mask.sum()/len(policy_actions)*100:.1f}%)")

# Calculate values
# Baseline: follow historical actions
baseline_value = rewards_test[actions_test == 1].mean()  # Only approved loans
baseline_total = rewards_test[actions_test == 1].sum()

# Policy: use RL decisions
policy_total = rewards_test[approve_mask].sum()
policy_value = policy_total / len(rewards_test)

improvement = policy_value - baseline_value
improvement_pct = (improvement / abs(baseline_value) * 100) if baseline_value != 0 else 0

print(f"\n💰 Financial Outcomes:")
print(f"  Baseline (historical): ${baseline_value:.2f} per loan")
print(f"  RL Policy:             ${policy_value:.2f} per loan")
print(f"  Improvement:           ${improvement:.2f} ({improvement_pct:+.1f}%)")

# Analysis of denied loans
if deny_mask.sum() > 0:
    denied_test_actions = actions_test[deny_mask]
    denied_rewards = rewards_test[deny_mask]
    
    # Of the loans RL denied, how many were originally approved (and how did they do)?
    originally_approved_but_denied = denied_test_actions == 1
    if originally_approved_but_denied.sum() > 0:
        avg_reward_if_approved = denied_rewards[originally_approved_but_denied].mean()
        print(f"\n🛑 Denied Loans Analysis:")
        print(f"  Originally approved but RL denies: {originally_approved_but_denied.sum():,}")
        print(f"  Avg reward if approved: ${avg_reward_if_approved:.2f}")
        if avg_reward_if_approved < 0:
            print(f"  ✓ Good! RL correctly identified risky loans")
        else:
            print(f"  ⚠ RL may be too conservative")

# ============================================================================
# 8: Visualization
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Policy actions
action_counts = [deny_mask.sum(), approve_mask.sum()]
axes[0,0].bar(['Deny', 'Approve'], action_counts, 
             color=['red', 'green'], alpha=0.7, edgecolor='black')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Fixed Policy Actions', fontweight='bold')
axes[0,0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(action_counts):
    axes[0,0].text(i, v, f'{v:,}\n({v/len(policy_actions)*100:.1f}%)', 
                  ha='center', va='bottom', fontweight='bold')

# 2. Training data composition
train_approve = (actions_train == 1).sum()
train_deny = (actions_train == 0).sum()
axes[0,1].bar(['Approve\n(Historical)', 'Deny\n(Synthetic)'], 
             [train_approve, train_deny],
             color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Training Data Composition', fontweight='bold')
axes[0,1].grid(True, alpha=0.3, axis='y')

# 3. Value comparison
values = [baseline_value, policy_value]
colors_bar = ['orange', 'blue']
bars = axes[1,0].bar(['Baseline', 'RL Policy'], values, 
                    color=colors_bar, alpha=0.7, edgecolor='black')
axes[1,0].set_ylabel('Avg Value per Loan ($)')
axes[1,0].set_title('Policy Value Comparison', fontweight='bold')
axes[1,0].axhline(0, color='r', linestyle='--', linewidth=1)
axes[1,0].grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    axes[1,0].text(bar.get_x() + bar.get_width()/2., height,
                  f'${val:.2f}', ha='center', 
                  va='bottom' if val > 0 else 'top',
                  fontweight='bold')

# 4. Summary
axes[1,1].axis('off')
summary_text = f"""
FIXED MODEL SUMMARY

Key Improvements:
• Added synthetic deny examples
• Balanced approve/deny ratio
• Higher alpha (conservative)
• Proper reward normalization

Results:
  Training time: {training_time/60:.1f} min
  Samples: {len(states_train):,}
  
Policy Actions:
  Deny:    {deny_mask.sum():,} ({deny_mask.sum()/len(policy_actions)*100:.1f}%)
  Approve: {approve_mask.sum():,} ({approve_mask.sum()/len(policy_actions)*100:.1f}%)

Performance:
  Baseline:    ${baseline_value:.2f}
  RL Policy:   ${policy_value:.2f}
  Improvement: ${improvement:.2f}
               ({improvement_pct:+.1f}%)

Status: {'✓ FIXED!' if deny_mask.sum() > 0 and improvement >= 0 else '⚠ Needs more tuning'}
"""
axes[1,1].text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
              verticalalignment='center')

plt.suptitle('Fixed Offline RL for Loan Approval', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fixed_rl_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualizations saved!\n")

# ============================================================================
# 9: Save Results
# ============================================================================
rl_results_fixed = {
    'policy_actions': policy_actions,
    'test_actions': actions_test,
    'rewards': rewards_test,
    'policy_value': policy_value,
    'baseline_value': baseline_value,
    'improvement': improvement,
    'improvement_pct': improvement_pct,
    'deny_count': deny_mask.sum(),
    'approve_count': approve_mask.sum()
}

with open('rl_results_fixed.pkl', 'wb') as f:
    pickle.dump(rl_results_fixed, f)

print("✓ Results saved: rl_results_fixed.pkl\n")

print("="*70)
print("RECOMMENDATIONS")
print("="*70)
print("\n1. If RL still approves everything (>95%):")
print("   • Increase alpha to 10.0")
print("   • Add more synthetic deny examples")
print("   • Use DL model predictions as a feature")
print("\n2. If RL denies everything:")
print("   • Decrease alpha to 2.0")
print("   • Check reward function scaling")
print("\n3. For production:")
print("   • Use hybrid: DL screens, RL decides")
print("   • A/B test with small percentage")
print("   • Monitor approval rates by demographics")
print("\n🚀 Fixed model complete!")