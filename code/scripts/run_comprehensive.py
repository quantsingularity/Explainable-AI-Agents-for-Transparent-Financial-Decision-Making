"""
Comprehensive experiment runner with all models, methods, and evaluation.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.fetch_data import DataFetcher
from agents.orchestrator import Orchestrator
from eval.metrics import XAIEvaluator, StatisticalTester
from eval.human_study import HumanStudySimulator, CounterfactualGenerator

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")

def run_comprehensive_experiment(mode: str = "full", seed: int = 42):
    """
    Run comprehensive experiment with:
    - Multiple models (logistic, tree, RF, GBM)
    - Multiple XAI methods (SHAP, LIME)
    - Comprehensive evaluation metrics
    - Statistical testing
    - Human study simulation
    - Counterfactual examples
    """
    
    start_time = time.time()
    
    logger.info("="*80)
    logger.info(f"COMPREHENSIVE EXPERIMENT - Mode: {mode}, Seed: {seed}")
    logger.info("="*80)
    
    # Setup results directory
    results_dir = f"results/{mode}_comprehensive"
    os.makedirs(results_dir, exist_ok=True)
    
    log_file = os.path.join(results_dir, "experiment.log")
    logger.add(log_file, level="DEBUG")
    
    # ========== STAGE 1: Data Preparation ==========
    logger.info("\n[STAGE 1/6] Data Preparation")
    data_fetcher = DataFetcher(seed=seed)
    df, feature_cols = data_fetcher.fetch_lending_data(mode=mode)
    X_train, X_test, y_train, y_test, scaler, feature_names = \
        data_fetcher.prepare_train_test_split(df, feature_cols)
    
    logger.info(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Class distribution: {np.bincount(y_train)}")
    
    # ========== STAGE 2: Model Training & Evaluation ==========
    logger.info("\n[STAGE 2/6] Model Training & Evaluation")
    
    # Model configurations
    if mode == "quick":
        model_configs = [
            ("logistic", "shap"),
            ("logistic", "lime"),
            ("tree", "shap"),
            ("tree", "lime")
        ]
    else:  # full
        model_configs = [
            ("logistic", "shap"),
            ("logistic", "lime"),
            ("tree", "shap"),
            ("tree", "lime"),
            ("random_forest", "shap"),
            ("random_forest", "lime"),
            ("gbm", "shap"),
            ("gbm", "lime")
        ]
    
    all_results = []
    all_explanations = []
    
    evaluator = XAIEvaluator(seed=seed)
    
    for model_type, xai_method in model_configs:
        logger.info(f"\n--- Experiment: {model_type} + {xai_method} ---")
        
        exp_start = time.time()
        
        # Train model
        orchestrator = Orchestrator(
            model_type=model_type,
            xai_method=xai_method,
            explanation_style="regulatory",
            seed=seed
        )
        
        orchestrator.train(X_train, y_train, feature_names)
        
        # Evaluate model performance
        y_pred, y_proba = orchestrator.decision_agent.predict(X_test)
        
        metrics = {
            'model_type': model_type,
            'xai_method': xai_method,
            'auc': roc_auc_score(y_test, y_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        logger.info(f"  Performance: AUC={metrics['auc']:.3f}, "
                   f"Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
        
        # Generate explanations
        n_explain = min(30, len(X_test)) if mode == "full" else min(10, len(X_test))
        explain_indices = np.random.RandomState(seed).choice(len(X_test), n_explain, replace=False)
        
        logger.info(f"  Generating {n_explain} explanations...")
        
        faithfulness_scores = []
        fidelity_scores = []
        completeness_scores = []
        explanation_times = []
        
        for idx in explain_indices:
            try:
                # Generate explanation
                explanation = orchestrator.explain_decision(X_test, idx)
                all_explanations.append(explanation)
                
                # Compute metrics
                xai_output = explanation['xai_explanation']
                attributions = xai_output['attributions']
                
                eval_metrics = evaluator.evaluate_comprehensive(
                    orchestrator.decision_agent.model,
                    X_test[idx:idx+1],
                    attributions,
                    feature_names
                )
                
                faithfulness_scores.append(eval_metrics['faithfulness'])
                fidelity_scores.append(eval_metrics['fidelity'])
                completeness_scores.append(eval_metrics['completeness'])
                explanation_times.append(explanation['metadata']['elapsed_time_seconds'])
                
            except Exception as e:
                logger.warning(f"  Failed on instance {idx}: {e}")
        
        # Aggregate XAI metrics
        metrics.update({
            'mean_faithfulness': np.mean(faithfulness_scores),
            'std_faithfulness': np.std(faithfulness_scores),
            'mean_fidelity': np.mean(fidelity_scores),
            'mean_completeness': np.mean(completeness_scores),
            'mean_explanation_time': np.mean(explanation_times),
            'std_explanation_time': np.std(explanation_times),
            'n_explanations': len(faithfulness_scores)
        })
        
        exp_time = time.time() - exp_start
        metrics['experiment_time_seconds'] = exp_time
        
        logger.info(f"  XAI Metrics: Faithfulness={metrics['mean_faithfulness']:.3f}, "
                   f"Fidelity={metrics['mean_fidelity']:.3f}, "
                   f"Completeness={metrics['mean_completeness']:.3f}")
        logger.info(f"  Time: {exp_time:.1f}s total, {metrics['mean_explanation_time']:.3f}s per explanation")
        
        all_results.append(metrics)
        
        # Save audit log
        audit_path = os.path.join(results_dir, f"audit_{model_type}_{xai_method}.jsonl")
        orchestrator.save_audit_log(audit_path)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(results_dir, "experiment_results.csv")
    results_df.to_csv(results_csv, index=False)
    
    logger.info(f"\n✓ Results saved to {results_csv}")
    
    # ========== STAGE 3: Statistical Testing ==========
    logger.info("\n[STAGE 3/6] Statistical Significance Testing")
    
    tester = StatisticalTester(seed=seed)
    
    # Compare SHAP vs LIME
    shap_results = results_df[results_df['xai_method'] == 'shap']
    lime_results = results_df[results_df['xai_method'] == 'lime']
    
    if len(shap_results) > 0 and len(lime_results) > 0:
        # Faithfulness comparison
        faith_test = tester.bootstrap_test(
            shap_results['mean_faithfulness'].values,
            lime_results['mean_faithfulness'].values,
            n_iterations=1000
        )
        
        logger.info(f"\nSHAP vs LIME Faithfulness:")
        logger.info(f"  Difference: {faith_test['observed_difference']:.4f}")
        logger.info(f"  95% CI: [{faith_test['ci_lower']:.4f}, {faith_test['ci_upper']:.4f}]")
        logger.info(f"  P-value: {faith_test['p_value']:.4f}")
        logger.info(f"  Significant: {faith_test['significant']}")
        
        # Save statistical tests
        stats_results = {
            'shap_vs_lime_faithfulness': faith_test
        }
        
        stats_path = os.path.join(results_dir, "statistical_tests.json")
        with open(stats_path, 'w') as f:
            json.dump(stats_results, f, indent=2)
    
    # ========== STAGE 4: Human Study Simulation ==========
    logger.info("\n[STAGE 4/6] Human Study Simulation")
    
    human_sim = HumanStudySimulator(seed=seed)
    human_df = human_sim.simulate_study(all_explanations, n_participants=120)
    
    human_csv = os.path.join(results_dir, "human_study_results.csv")
    human_df.to_csv(human_csv, index=False)
    logger.info(f"✓ Human study results saved to {human_csv}")
    
    # Compute effect sizes
    effect_sizes = human_sim.compute_effect_sizes(human_df)
    logger.info(f"\nEffect Sizes (Narrative vs No Explanation):")
    logger.info(f"  Trust: Cohen's d = {effect_sizes['trust_cohens_d']:.3f}, p = {effect_sizes['trust_p_value']:.4f}")
    logger.info(f"  Accuracy: Cohen's d = {effect_sizes['accuracy_cohens_d']:.3f}, p = {effect_sizes['accuracy_p_value']:.4f}")
    
    # Save effect sizes
    effect_path = os.path.join(results_dir, "effect_sizes.json")
    with open(effect_path, 'w') as f:
        json.dump(effect_sizes, f, indent=2)
    
    # ========== STAGE 5: Counterfactual Examples ==========
    logger.info("\n[STAGE 5/6] Generating Counterfactual Examples")
    
    cf_gen = CounterfactualGenerator(seed=seed)
    counterfactuals = []
    
    # Generate counterfactuals for 5 denied applications
    denied_indices = np.where(y_test == 0)[0][:5]
    
    # Use logistic model for counterfactuals
    orchestrator = Orchestrator(model_type="logistic", xai_method="shap", seed=seed)
    orchestrator.train(X_train, y_train, feature_names)
    
    for idx in denied_indices:
        try:
            cf = cf_gen.generate_counterfactual(
                orchestrator.decision_agent.model,
                X_test[idx],
                feature_names,
                desired_class=1,
                max_changes=3
            )
            counterfactuals.append(cf)
            
            logger.info(f"  Counterfactual {idx}: {len(cf['changes'])} changes needed")
            
        except Exception as e:
            logger.warning(f"  Failed counterfactual {idx}: {e}")
    
    cf_path = os.path.join(results_dir, "counterfactuals.json")
    with open(cf_path, 'w') as f:
        json.dump(counterfactuals, f, indent=2, default=str)
    
    logger.info(f"✓ {len(counterfactuals)} counterfactuals saved")
    
    # ========== STAGE 6: Summary Report ==========
    logger.info("\n[STAGE 6/6] Summary Report")
    
    total_time = time.time() - start_time
    
    summary = {
        'experiment_mode': mode,
        'seed': seed,
        'total_runtime_seconds': total_time,
        'total_runtime_hours': total_time / 3600,
        'n_experiments': len(all_results),
        'n_explanations_generated': sum(r['n_explanations'] for r in all_results),
        'best_auc': results_df['auc'].max(),
        'best_model': results_df.loc[results_df['auc'].idxmax(), 'model_type'],
        'best_faithfulness': results_df['mean_faithfulness'].max(),
        'best_xai_method': results_df.loc[results_df['mean_faithfulness'].idxmax(), 'xai_method'],
        'dataset_size': len(X_train) + len(X_test),
        'n_features': len(feature_names),
        'human_study_participants': len(human_df),
        'n_counterfactuals': len(counterfactuals)
    }
    
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Total Runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logger.info(f"Experiments: {summary['n_experiments']}")
    logger.info(f"Explanations: {summary['n_explanations_generated']}")
    logger.info(f"Best AUC: {summary['best_auc']:.3f} ({summary['best_model']})")
    logger.info(f"Best Faithfulness: {summary['best_faithfulness']:.3f} ({summary['best_xai_method']})")
    logger.info(f"\nResults directory: {results_dir}")
    logger.info("="*80)
    
    return results_df, summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["quick", "full"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    results, summary = run_comprehensive_experiment(args.mode, args.seed)
    
    print("\n✅ All experiments complete! Check results directory for outputs.")
