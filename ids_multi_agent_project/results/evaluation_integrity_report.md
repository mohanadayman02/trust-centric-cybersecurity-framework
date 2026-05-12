# Evaluation Integrity Report

- Test samples: 44556
- Test class distribution: normal=23117, attack=21439
- Trust scores are derived from validation metrics only; test labels are only used for final evaluation.
- All reported metrics use the full test set. No reported row is computed on a filtered subset.

## Prediction Distribution
- Stage 1: Basic Feature Model: attack_rate=0.4754, normal_rate=0.5246, almost_all_attack=False, almost_all_normal=False
- Stage 2: Content Feature Model: attack_rate=0.6102, normal_rate=0.3898, almost_all_attack=False, almost_all_normal=False
- Stage 3: Time Traffic Feature Model: attack_rate=0.4410, normal_rate=0.5590, almost_all_attack=False, almost_all_normal=False
- Stage 4: Host Traffic Feature Model: attack_rate=0.4705, normal_rate=0.5295, almost_all_attack=False, almost_all_normal=False
- Logistic Regression Model: attack_rate=0.4850, normal_rate=0.5150, almost_all_attack=False, almost_all_normal=False
- ExtraTrees Model: attack_rate=0.4815, normal_rate=0.5185, almost_all_attack=False, almost_all_normal=False
- HistGradientBoosting Model: attack_rate=0.4814, normal_rate=0.5186, almost_all_attack=False, almost_all_normal=False
- majority_voting: attack_rate=0.4832, normal_rate=0.5168, almost_all_attack=False, almost_all_normal=False
- accuracy_based_trust: attack_rate=0.4832, normal_rate=0.5168, almost_all_attack=False, almost_all_normal=False
- f1_based_trust: attack_rate=0.4832, normal_rate=0.5168, almost_all_attack=False, almost_all_normal=False
- error_aware_trust: attack_rate=0.4832, normal_rate=0.5168, almost_all_attack=False, almost_all_normal=False
- confidence_based_trust: attack_rate=0.4764, normal_rate=0.5236, almost_all_attack=False, almost_all_normal=False
- confidence_margin_trust: attack_rate=0.4844, normal_rate=0.5156, almost_all_attack=False, almost_all_normal=False
- class_specific_trust: attack_rate=0.4832, normal_rate=0.5168, almost_all_attack=False, almost_all_normal=False
- dynamic_trust: attack_rate=0.4832, normal_rate=0.5168, almost_all_attack=False, almost_all_normal=False
- hybrid_trust: attack_rate=0.4764, normal_rate=0.5236, almost_all_attack=False, almost_all_normal=False
- attack_recall_trust: attack_rate=0.4832, normal_rate=0.5168, almost_all_attack=False, almost_all_normal=False
- fnr_penalty_trust: attack_rate=0.4832, normal_rate=0.5168, almost_all_attack=False, almost_all_normal=False
- attack_override_trust: attack_rate=0.6377, normal_rate=0.3623, almost_all_attack=False, almost_all_normal=False
- best_safe_model_selector: attack_rate=0.4814, normal_rate=0.5186, almost_all_attack=False, almost_all_normal=False
- best_accuracy_selector: attack_rate=0.4814, normal_rate=0.5186, almost_all_attack=False, almost_all_normal=False
- local_accuracy_trust: attack_rate=0.4831, normal_rate=0.5169, almost_all_attack=False, almost_all_normal=False
- stacking_meta_trust: attack_rate=0.4813, normal_rate=0.5187, almost_all_attack=False, almost_all_normal=False

- No method in the current run is evaluated on a filtered subset.