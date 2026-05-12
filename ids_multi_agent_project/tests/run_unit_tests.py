import importlib.util
import pathlib
import sys

# ensure project root is on sys.path so imports like 'pipeline' resolve
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

spec = importlib.util.spec_from_file_location(
    "test_trust_methods",
    str(pathlib.Path(__file__).parent / "test_trust_methods.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

report_spec = importlib.util.spec_from_file_location(
    "test_reporting_utils",
    str(pathlib.Path(__file__).parent / "test_reporting_utils.py"),
)
report_module = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(report_module)

four_agent_spec = importlib.util.spec_from_file_location(
    "test_four_agent_mode",
    str(pathlib.Path(__file__).parent / "test_four_agent_mode.py"),
)
four_agent_module = importlib.util.module_from_spec(four_agent_spec)
four_agent_spec.loader.exec_module(four_agent_module)

ton_spec = importlib.util.spec_from_file_location(
    "test_ton_iot_dataset",
    str(pathlib.Path(__file__).parent / "test_ton_iot_dataset.py"),
)
ton_module = importlib.util.module_from_spec(ton_spec)
ton_spec.loader.exec_module(ton_module)

cic_spec = importlib.util.spec_from_file_location(
    "test_cicids2017_dataset",
    str(pathlib.Path(__file__).parent / "test_cicids2017_dataset.py"),
)
cic_module = importlib.util.module_from_spec(cic_spec)
cic_spec.loader.exec_module(cic_module)

funcs = [
    module.test_majority_voting_simple,
    module.test_accuracy_based_trust_prefers_best,
    module.test_error_aware_punishes_fnr,
    module.test_confidence_based_trust_handles_missing_probs,
    module.test_class_specific_trust_returns_array,
    module.test_hybrid_trust_outputs_weights,
    module.test_attack_recall_trust_prefers_high_recall_model,
    module.test_fnr_penalty_trust_penalizes_high_fnr,
    module.test_attack_override_trust_chooses_attack_for_confident_trusted_model,
    module.test_best_safe_model_selector_picks_highest_safety_score,
    module.test_role_aware_trust_voting_handles_disagreement_cases,
    module.test_trust_agent_selector_returns_prediction_per_sample,
    module.test_trust_agent_selector_uses_disagreement_bonus_for_hard_case,
    report_module.test_oracle_upper_bound_computation,
    report_module.test_model_diversity_counts_are_correct,
    report_module.test_confidence_margin_trust_prefers_high_confidence_model,
    report_module.test_best_accuracy_selector_chooses_highest_validation_accuracy,
    report_module.test_stacking_meta_trust_signature_avoids_test_labels,
    four_agent_module.test_exactly_four_agents_in_strict_mode_definition,
    four_agent_module.test_attack_threshold_tuning_improves_recall_vs_default,
    four_agent_module.test_normal_threshold_tuning_improves_specificity_vs_default,
    four_agent_module.test_threshold_tuning_uses_only_given_validation_labels,
    four_agent_module.test_identify_hard_validation_cases_marks_disagreement,
    four_agent_module.test_identify_hard_validation_cases_marks_low_confidence,
    four_agent_module.test_general_agent_view_selection_uses_target_range_on_validation,
    ton_module.test_ton_loader_detects_binary_label_column,
    ton_module.test_ton_loader_drops_leakage_columns,
    ton_module.test_ton_loader_categorical_columns_encodeable,
    ton_module.test_ton_loader_split_shapes,
    ton_module.test_ton_dataset_aliases_supported,
    cic_module.test_cic_loader_detects_label_and_maps_binary,
    cic_module.test_cic_loader_drops_leakage_columns,
    cic_module.test_cic_loader_handles_inf_values,
    cic_module.test_cic_loader_prefers_processed_train_test,
    cic_module.test_cic_loader_single_csv_fallback,
    cic_module.test_cic_alias_and_output_directory_name,
]

failed = []
for f in funcs:
    try:
        f()
        print(f"OK: {f.__name__}")
    except Exception as e:
        print(f"FAIL: {f.__name__} -> {e}")
        failed.append((f.__name__, str(e)))

if failed:
    print(f"{len(failed)} test(s) failed")
    raise SystemExit(1)
print("All trust method tests passed")
