# ============================================================================
# STEP 5: COMPREHENSIVE ANALYSIS & COMPARISON REPORT
# Task 4: Analysis, Comparison, and Future Steps
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 5: DL vs RL POLICY COMPARISON & ANALYSIS REPORT")
print("="*80)

# ============================================================================
# 1: LOAD ALL ARTIFACTS
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: LOADING ARTIFACTS")
print("="*80)

# Load DL results
try:
    with open('dl_results.pkl', 'rb') as f:
        dl_results = pickle.load(f)
    print("✓ DL results loaded")
except:
    print("✗ dl_results.pkl not found")
    exit(1)

# Load DL history
try:
    with open('dl_history.pkl', 'rb') as f:
        dl_history = pickle.load(f)
    print("✓ DL training history loaded")
except:
    print("! DL history not found (optional)")
    dl_history = None

# Load RL results (try both filenames)
rl_results = None
rl_source = None
for fname in ['rl_results_fixed.pkl', 'rl_results_fast.pkl', 'rl_results.pkl']:
    try:
        with open(fname, 'rb') as f:
            rl_results = pickle.load(f)
        rl_source = fname
        print(f"✓ RL results loaded from: {fname}")
        break
    except:
        continue

if rl_results is None:
    print("✗ No RL results file found")
    exit(1)

# Load preprocessed data for context
try:
    df_data = pd.read_pickle('df_preprocessed.pkl')
    print(f"✓ Preprocessed data loaded: {df_data.shape}")
except:
    print("! Preprocessed data not found (optional)")
    df_data = None

# ============================================================================
# 2: EXTRACT & SUMMARIZE DL METRICS
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: DEEP LEARNING MODEL METRICS")
print("="*80)

# DL model evaluation metrics
y_true_dl = dl_results['y_true']
y_pred_proba_dl = dl_results['y_pred_proba']
y_pred_dl = (y_pred_proba_dl >= 0.5).astype(int)  # Default threshold

# Calculate metrics
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

auc_score = roc_auc_score(y_true_dl, y_pred_proba_dl)
f1_default = f1_score(y_true_dl, y_pred_dl)
precision_default = precision_score(y_true_dl, y_pred_dl)
recall_default = recall_score(y_true_dl, y_pred_dl)

# Find optimal threshold (maximize F1)
fpr, tpr, thresholds = roc_curve(y_true_dl, y_pred_proba_dl)
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true_dl, y_pred_proba_dl)

f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else 0.5
optimal_f1 = f1_scores[optimal_idx]

y_pred_optimal = (y_pred_proba_dl >= optimal_threshold).astype(int)
f1_optimal = f1_score(y_true_dl, y_pred_optimal)
precision_optimal = precision_score(y_true_dl, y_pred_optimal)
recall_optimal = recall_score(y_true_dl, y_pred_optimal)

print("\n📊 DL MODEL PERFORMANCE")
print(f"\n  ROC-AUC Score:              {auc_score:.4f}")
print(f"\n  Threshold: 0.50 (default)")
print(f"    • F1-Score:               {f1_default:.4f}")
print(f"    • Precision:              {precision_default:.4f}")
print(f"    • Recall:                 {recall_default:.4f}")
print(f"\n  Threshold: {optimal_threshold:.4f} (optimal for F1)")
print(f"    • F1-Score:               {f1_optimal:.4f}")
print(f"    • Precision:              {precision_optimal:.4f}")
print(f"    • Recall:                 {recall_optimal:.4f}")

# Approval rates
approval_rate_default = (y_pred_dl == 0).sum() / len(y_pred_dl) * 100
approval_rate_optimal = (y_pred_optimal == 0).sum() / len(y_pred_optimal) * 100

print(f"\n  Approval Rates:")
print(f"    • At threshold 0.50:      {approval_rate_default:.1f}%")
print(f"    • At threshold {optimal_threshold:.4f}: {approval_rate_optimal:.1f}%")

# Use optimal for downstream
y_pred_dl_policy = y_pred_optimal
dl_threshold = optimal_threshold

print(f"\n✓ Using optimal threshold {optimal_threshold:.4f} for DL policy")

# ============================================================================
# 5.2: Extract Results for Comparison
# ============================================================================
print("="*70)
print("PREPARING DATA FOR COMPARISON")
print("="*70)

# DL predictions
dl_proba = dl_results['y_pred_proba']
dl_pred = dl_results['y_pred']  # 1 = high risk (default), 0 = low risk (paid)
y_true = dl_results['y_true']

# RL predictions
rl_actions = rl_results['policy_actions']  # 1 = approve, 0 = deny
rewards = rl_results.get('rewards_original', rl_results.get('rewards', np.zeros(len(rl_actions))))
actual_outcomes = rl_results['actual_outcomes']

# Note: DL predicts DEFAULT probability, so we need to convert to approval decision
# DL approval decision: approve if default probability < 0.5
dl_approve = (dl_proba < 0.5).astype(int)

# Align to the smaller dataset (RL test set)
min_len = min(len(dl_proba), len(rl_actions))
dl_proba = dl_proba[:min_len]
dl_pred = dl_pred[:min_len]
y_true = y_true[:min_len]
dl_approve = dl_approve[:min_len]
rl_actions = rl_actions[:min_len]
rewards = rewards[:min_len]
actual_outcomes = actual_outcomes[:min_len]

print(f"\nData prepared (aligned to {min_len:,} samples):")
print(f"  Test set size: {len(dl_proba):,}")
print(f"  DL approves: {dl_approve.sum():,} ({dl_approve.mean()*100:.2f}%)")
print(f"  RL approves: {rl_actions.sum():,} ({rl_actions.mean()*100:.2f}%)")
print()

# ============================================================================
# 5.3: Create Comparison DataFrame
# ============================================================================
print("="*70)
print("CREATING COMPARISON DATAFRAME")
print("="*70)

comparison_df = pd.DataFrame({
    'dl_default_prob': dl_proba,      # DL's predicted default probability
    'dl_approve': dl_approve,          # DL's approval decision (0=deny, 1=approve)
    'rl_approve': rl_actions,          # RL's approval decision (0=deny, 1=approve)
    'actual_default': actual_outcomes, # Actual outcome (0=paid, 1=default)
    'reward': rewards                  # Actual reward if approved
})

print("✓ Comparison dataframe created")
print(f"\nDataframe shape: {comparison_df.shape}")
print("\nFirst few rows:")
print(comparison_df.head(10))
print()

# ============================================================================
# 5.4: Overall Agreement Analysis
# ============================================================================
print("="*70)
print("DECISION AGREEMENT ANALYSIS")
print("="*70)

# Calculate agreement
comparison_df['models_agree'] = (comparison_df['dl_approve'] == comparison_df['rl_approve'])
agreement_rate = comparison_df['models_agree'].mean()

print(f"\nOverall Agreement: {agreement_rate*100:.2f}%")
print(f"Disagreement: {(1-agreement_rate)*100:.2f}%")

# Decision matrix
print("\n" + "-"*70)
print("Decision Matrix (DL vs RL):")
print("-"*70)
decision_matrix = pd.crosstab(
    comparison_df['dl_approve'], 
    comparison_df['rl_approve'],
    rownames=['DL Model'],
    colnames=['RL Agent'],
    margins=True
)
print(decision_matrix)
print()

# Analyze each quadrant
both_deny = ((comparison_df['dl_approve'] == 0) & (comparison_df['rl_approve'] == 0)).sum()
both_approve = ((comparison_df['dl_approve'] == 1) & (comparison_df['rl_approve'] == 1)).sum()
dl_deny_rl_approve = ((comparison_df['dl_approve'] == 0) & (comparison_df['rl_approve'] == 1)).sum()
dl_approve_rl_deny = ((comparison_df['dl_approve'] == 1) & (comparison_df['rl_approve'] == 0)).sum()

print(f"\n" + "-"*70)
print("Decision Breakdown:")
print("-"*70)
print(f"Both models DENY:        {both_deny:,} ({both_deny/len(comparison_df)*100:.2f}%)")
print(f"Both models APPROVE:     {both_approve:,} ({both_approve/len(comparison_df)*100:.2f}%)")
print(f"DL denies, RL approves:  {dl_deny_rl_approve:,} ({dl_deny_rl_approve/len(comparison_df)*100:.2f}%)")
print(f"DL approves, RL denies:  {dl_approve_rl_deny:,} ({dl_approve_rl_deny/len(comparison_df)*100:.2f}%)")
print()

# ============================================================================
# 5.5: CASE 1 - High Risk but RL Approves
# ============================================================================
print("="*70)
print("CASE 1: HIGH RISK APPLICANT, BUT RL APPROVES")
print("="*70)
print("DL says HIGH default risk (>70%), but RL agent APPROVES")

high_risk_approved = comparison_df[
    (comparison_df['dl_default_prob'] > 0.7) &  # High default risk
    (comparison_df['rl_approve'] == 1)           # RL approves
].copy()

print(f"\nFound {len(high_risk_approved):,} cases ({len(high_risk_approved)/len(comparison_df)*100:.2f}%)")

if len(high_risk_approved) > 0:
    print("\n" + "-"*70)
    print("Sample Cases (first 10):")
    print("-"*70)
    print(high_risk_approved.head(10).to_string(index=False))
    
    print("\n" + "-"*70)
    print("Analysis:")
    print("-"*70)
    print(f"  Average default probability (DL): {high_risk_approved['dl_default_prob'].mean()*100:.2f}%")
    print(f"  Actual default rate: {high_risk_approved['actual_default'].mean()*100:.2f}%")
    print(f"  Average reward: ${high_risk_approved['reward'].mean():.2f}")
    print(f"  Total reward: ${high_risk_approved['reward'].sum():,.2f}")
    
    # Get interest rates for these loans (from original data)
    high_risk_indices = high_risk_approved.index
    if len(rl_results.get('test_indices', [])) > 0:
        try:
            high_risk_int_rates = df_features.iloc[rl_results['test_indices'][high_risk_indices]]['int_rate']
            print(f"  Average interest rate: {high_risk_int_rates.mean():.2f}%")
        except:
            pass
    
    print("\n💡 WHY MIGHT RL APPROVE HIGH-RISK LOANS?")
    print("   → High interest rates can compensate for default risk")
    print("   → RL optimizes for EXPECTED VALUE, not just accuracy")
    print("   → Some high-risk loans are still profitable")
    print("   → Risk-adjusted returns matter more than default probability")
else:
    print("\n⚠ No cases found where RL approves high-risk applicants")

print()

# ============================================================================
# 5.6: CASE 2 - Low Risk but RL Denies
# ============================================================================
print("="*70)
print("CASE 2: LOW RISK APPLICANT, BUT RL DENIES")
print("="*70)
print("DL says LOW default risk (<30%), but RL agent DENIES")

low_risk_denied = comparison_df[
    (comparison_df['dl_default_prob'] < 0.3) &  # Low default risk
    (comparison_df['rl_approve'] == 0)           # RL denies
].copy()

print(f"\nFound {len(low_risk_denied):,} cases ({len(low_risk_denied)/len(comparison_df)*100:.2f}%)")

if len(low_risk_denied) > 0:
    print("\n" + "-"*70)
    print("Sample Cases (first 10):")
    print("-"*70)
    print(low_risk_denied.head(10).to_string(index=False))
    
    print("\n" + "-"*70)
    print("Analysis:")
    print("-"*70)
    print(f"  Average default probability (DL): {low_risk_denied['dl_default_prob'].mean()*100:.2f}%")
    print(f"  Actual default rate: {low_risk_denied['actual_default'].mean()*100:.2f}%")
    print(f"  Average Q-value (RL): ${low_risk_denied['rl_q_value'].mean():.2f}")
    print(f"  Average potential reward: ${low_risk_denied['reward'].mean():.2f}")
    
    # Get interest rates
    low_risk_indices = low_risk_denied.index
    low_risk_int_rates = df_features.iloc[rl_results['test_indices'][low_risk_indices]]['int_rate']
    print(f"  Average interest rate: {low_risk_int_rates.mean():.2f}%")
    
    print("\n💡 WHY MIGHT RL DENY LOW-RISK LOANS?")
    print("   → Low interest rates = low potential reward")
    print("   → Opportunity cost: capital could be better deployed elsewhere")
    print("   → Risk-adjusted return not attractive enough")
    print("   → RL optimizes for VALUE, not just low default rate")
else:
    print("\n⚠ No cases found where RL denies low-risk applicants")

print()

# ============================================================================
# 5.7: Analyze Performance When Models Agree vs Disagree
# ============================================================================
print("="*70)
print("PERFORMANCE WHEN MODELS AGREE VS DISAGREE")
print("="*70)

# When models agree
agree_cases = comparison_df[comparison_df['models_agree'] == True]
disagree_cases = comparison_df[comparison_df['models_agree'] == False]

print("\nWhen Models AGREE:")
print(f"  Cases: {len(agree_cases):,}")
print(f"  Actual default rate: {agree_cases['actual_default'].mean()*100:.2f}%")
print(f"  Average reward: ${agree_cases['reward'].mean():.2f}")

print("\nWhen Models DISAGREE:")
print(f"  Cases: {len(disagree_cases):,}")
print(f"  Actual default rate: {disagree_cases['actual_default'].mean()*100:.2f}%")
print(f"  Average reward: ${disagree_cases['reward'].mean():.2f}")

# Break down disagreement cases
if len(disagree_cases) > 0:
    print("\n" + "-"*70)
    print("Disagreement Breakdown:")
    print("-"*70)
    
    dl_deny_rl_approve_cases = disagree_cases[
        (disagree_cases['dl_approve'] == 0) & (disagree_cases['rl_approve'] == 1)
    ]
    dl_approve_rl_deny_cases = disagree_cases[
        (disagree_cases['dl_approve'] == 1) & (disagree_cases['rl_approve'] == 0)
    ]
    
    if len(dl_deny_rl_approve_cases) > 0:
        print(f"\nDL denies, RL approves ({len(dl_deny_rl_approve_cases):,} cases):")
        print(f"  Avg default prob: {dl_deny_rl_approve_cases['dl_default_prob'].mean()*100:.2f}%")
        print(f"  Actual default rate: {dl_deny_rl_approve_cases['actual_default'].mean()*100:.2f}%")
        print(f"  Average reward: ${dl_deny_rl_approve_cases['reward'].mean():.2f}")
    
    if len(dl_approve_rl_deny_cases) > 0:
        print(f"\nDL approves, RL denies ({len(dl_approve_rl_deny_cases):,} cases):")
        print(f"  Avg default prob: {dl_approve_rl_deny_cases['dl_default_prob'].mean()*100:.2f}%")
        print(f"  Actual default rate: {dl_approve_rl_deny_cases['actual_default'].mean()*100:.2f}%")
        print(f"  Average potential reward: ${dl_approve_rl_deny_cases['reward'].mean():.2f}")

print()

# ============================================================================
# 5.8: Comprehensive Visualizations
# ============================================================================
print("="*70)
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. DL Probability vs RL Decision
ax1 = fig.add_subplot(gs[0, 0])
colors = ['red' if x == 0 else 'green' for x in rl_actions]
ax1.scatter(dl_proba, rl_actions, alpha=0.3, c=colors, s=5)
ax1.axvline(0.5, color='blue', linestyle='--', linewidth=2, label='DL Threshold')
ax1.set_xlabel('DL Default Probability', fontsize=10)
ax1.set_ylabel('RL Decision', fontsize=10)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Deny', 'Approve'])
ax1.set_title('DL Risk vs RL Decision', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Decision Agreement Heatmap
ax2 = fig.add_subplot(gs[0, 1])
decision_counts = pd.crosstab(comparison_df['dl_approve'], comparison_df['rl_approve'])
sns.heatmap(decision_counts, annot=True, fmt='d', cmap='YlOrRd',
           xticklabels=['Deny', 'Approve'],
           yticklabels=['Deny', 'Approve'],
           ax=ax2, cbar_kws={'label': 'Count'})
ax2.set_xlabel('RL Decision', fontsize=10)
ax2.set_ylabel('DL Decision', fontsize=10)
ax2.set_title('Decision Agreement Matrix', fontsize=11, fontweight='bold')

# 3. Reward by Agreement
ax3 = fig.add_subplot(gs[0, 2])
reward_by_agreement = comparison_df.groupby('models_agree')['reward'].mean()
bars = ax3.bar(['Disagree', 'Agree'], reward_by_agreement.values,
              color=['orange', 'blue'], alpha=0.7, edgecolor='black')
ax3.axhline(0, color='r', linestyle='--', linewidth=1)
ax3.set_ylabel('Average Reward ($)', fontsize=10)
ax3.set_title('Reward: Agreement vs Disagreement', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=9)

# 4. Default Rate by Decision Type
ax4 = fig.add_subplot(gs[1, 0])
decision_types = ['Both Deny', 'DL Deny\nRL Approve', 'DL Approve\nRL Deny', 'Both Approve']
default_rates = [
    comparison_df[(comparison_df['dl_approve']==0) & (comparison_df['rl_approve']==0)]['actual_default'].mean()*100,
    comparison_df[(comparison_df['dl_approve']==0) & (comparison_df['rl_approve']==1)]['actual_default'].mean()*100,
    comparison_df[(comparison_df['dl_approve']==1) & (comparison_df['rl_approve']==0)]['actual_default'].mean()*100,
    comparison_df[(comparison_df['dl_approve']==1) & (comparison_df['rl_approve']==1)]['actual_default'].mean()*100
]
ax4.bar(range(len(decision_types)), default_rates, color=['red', 'orange', 'yellow', 'green'], 
       alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(decision_types)))
ax4.set_xticklabels(decision_types, rotation=0, fontsize=8)
ax4.set_ylabel('Default Rate (%)', fontsize=10)
ax4.set_title('Actual Default Rate by Decision Pattern', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Reward Distribution by Decision Type
ax5 = fig.add_subplot(gs[1, 1])
both_approve_rewards = comparison_df[
    (comparison_df['dl_approve']==1) & (comparison_df['rl_approve']==1)
]['reward']
dl_deny_rl_approve_rewards = comparison_df[
    (comparison_df['dl_approve']==0) & (comparison_df['rl_approve']==1)
]['reward']
ax5.hist([both_approve_rewards, dl_deny_rl_approve_rewards], bins=50, 
        label=['Both Approve', 'DL Deny, RL Approve'], 
        alpha=0.7, edgecolor='black')
ax5.axvline(0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Reward ($)', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Reward Distribution by Decision', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# 6. Q-Value vs Default Probability (skipped - no Q values in RL results)
# Using reward instead
ax6 = fig.add_subplot(gs[1, 2])
scatter = ax6.scatter(dl_proba, rewards, c=actual_outcomes, 
                     cmap='RdYlGn_r', alpha=0.5, s=10)
ax6.axvline(0.5, color='blue', linestyle='--', linewidth=1, label='DL Threshold')
ax6.axhline(0, color='red', linestyle='--', linewidth=1, label='Break-even')
ax6.set_xlabel('DL Default Probability', fontsize=10)
ax6.set_ylabel('Actual Reward ($)', fontsize=10)
ax6.set_title('Reward vs Default Probability', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax6, label='Actual (0=Paid, 1=Default)')

# 7. Model Performance Comparison
ax7 = fig.add_subplot(gs[2, 0])
metrics = ['DL Model\n(AUC)', 'DL Model\n(F1)', 'RL Agent\n(Policy Value)']
# Get RL policy value - try different possible keys
rl_value = rl_results.get('estimated_policy_value', 
                          rl_results.get('policy_value', 
                          rl_results.get('avg_reward', comparison_df['reward'].mean())))
values = [dl_results['auc'], dl_results['f1'], rl_value/1000]
colors_bar = ['blue', 'blue', 'green']
bars = ax7.bar(range(len(metrics)), values, color=colors_bar, alpha=0.7, edgecolor='black')
ax7.set_xticks(range(len(metrics)))
ax7.set_xticklabels(metrics, fontsize=9)
ax7.set_ylabel('Value', fontsize=10)
ax7.set_title('Model Performance Metrics\n(RL value in $1000s)', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, values)):
    if i < 2:
        label = f'{val:.4f}'
    else:
        label = f'${val*1000:.0f}'
    ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            label, ha='center', va='bottom', fontsize=9, fontweight='bold')

# 8. Approval Rate Comparison
ax8 = fig.add_subplot(gs[2, 1])
approval_data = pd.DataFrame({
    'Model': ['DL Model', 'RL Agent', 'Historical'],
    'Approval Rate': [dl_approve.mean()*100, rl_actions.mean()*100, 100]
})
bars = ax8.bar(approval_data['Model'], approval_data['Approval Rate'],
              color=['blue', 'green', 'gray'], alpha=0.7, edgecolor='black')
ax8.set_ylabel('Approval Rate (%)', fontsize=10)
ax8.set_title('Loan Approval Rates', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 9. Expected Value by Policy
ax9 = fig.add_subplot(gs[2, 2])
# Get baseline and policy values with fallbacks
baseline_val = rl_results.get('baseline_value', comparison_df['reward'].mean())
policy_val = rl_results.get('estimated_policy_value',
                            rl_results.get('policy_value',
                            rl_results.get('avg_reward', comparison_df['reward'].mean())))
policy_values = pd.DataFrame({
    'Policy': ['Historical\n(Approve All)', 'RL Agent\nPolicy'],
    'Expected Value': [baseline_val, policy_val]
})
bars = ax9.bar(policy_values['Policy'], policy_values['Expected Value'],
              color=['gray', 'green'], alpha=0.7, edgecolor='black')
ax9.axhline(0, color='r', linestyle='--', linewidth=1)
ax9.set_ylabel('Expected Value ($)', fontsize=10)
ax9.set_title('Expected Value Comparison', fontsize=11, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=9, fontweight='bold')

plt.suptitle('Comprehensive Model Comparison Analysis', 
            fontsize=14, fontweight='bold', y=0.995)
plt.show()

print("✓ Visualizations generated!\n")

# ============================================================================
# 5.9: Key Insights and Recommendations
# ============================================================================
print("="*70)
print("KEY INSIGHTS AND RECOMMENDATIONS")
print("="*70)

print("\n📊 METRIC DIFFERENCES:")
print("-" *70)
print("\n1. Why AUC and F1 for DL Model?")
print("   → AUC measures discrimination ability (separating defaults from non-defaults)")
print("   → F1 balances precision and recall for imbalanced classification")
print("   → Both focus on PREDICTION ACCURACY, not financial outcomes")

print("\n2. Why Policy Value for RL Agent?")
print("   → Policy Value measures expected financial return")
print("   → Directly aligns with business objective: maximize profit")
print("   → Accounts for both risk AND reward")

print("\n\n💡 MODEL BEHAVIORS:")
print("-"*70)
print("\n1. DL Model (Risk Classifier):")
print("   → Predicts probability of default")
print(f"   → Approval rate: {dl_approve.mean()*100:.2f}%")
print("   → Focuses on minimizing misclassification")
print("   → Does NOT consider interest rates or loan amounts")

print("\n2. RL Agent (Policy Learner):")
print("   → Learns optimal approval policy")
print(f"   → Approval rate: {rl_actions.mean()*100:.2f}%")
print("   → Maximizes expected financial value")
print("   → Considers risk-reward tradeoff explicitly")

if len(high_risk_approved) > 0:
    print("\n3. Key Difference:")
    print("   → RL sometimes approves HIGH-RISK loans with high interest rates")
    print("   → Expected value can be positive even with higher default risk")

print("\n\n🎯 DEPLOYMENT RECOMMENDATIONS:")
print("-"*70)
print("\n1. Which model to deploy?")
print("   → Depends on business objectives")
print("   → If goal is profit maximization: RL Agent")
print("   → If goal is risk minimization: DL Model")
print("   → Consider hybrid approach for balanced strategy")

print("\n2. Hybrid Approach:")
print("   → Use DL for initial screening")
print("   → Use RL for final approval decision")
print("   → Combine risk assessment with value optimization")

print("\n\n⚠️  LIMITATIONS:")
print("-"*70)
print("\n1. Data Limitations:")
print("   → Only observed approved loans (selection bias)")
print("   → No data on denied loans' counterfactual outcomes")
print("   → Simplified reward function")

print("\n2. Model Limitations:")
print("   → RL trained only on historical approval policy")
print("   → May overestimate value for deny actions")
print("   → Need more sophisticated offline RL techniques")

print("\n3. Real-world Considerations:")
print("   → Regulatory compliance not modeled")
print("   → Fairness and bias not addressed")
print("   → Economic conditions change over time")

print("\n\n🔮 FUTURE STEPS:")
print("-"*70)
print("\n1. Data Collection:")
print("   → Collect data on denied loans (via randomized trials)")
print("   → Include macroeconomic indicators")
print("   → Track partial payments and recoveries")

print("\n2. Advanced Modeling:")
print("   → Try other offline RL algorithms (IQL, CQL variants)")
print("   → Incorporate confid ence bounds (Conservative policies)")
print("   → Multi-objective optimization (profit + fairness)")

print("\n3. Deployment Strategy:")
print("   → A/B testing with current policy")
print("   → Gradual rollout with monitoring")
print("   → Regular model retraining")

print("\n\n" + "="*70)
print("STEP 5 COMPLETE!")
print("All analysis finished! Check the generated visualizations and insights.")
print("="*70)