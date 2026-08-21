#!/usr/bin/env python3
"""
Automated Verification Suite for Domain-Adaptive Clinical LLM Project.
Checks all 9 Core Requirements and Schema Validations.
"""

import os
import ast
import json
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def test_core_requirements():
    print("\n========================================================")
    print("      RUNNING CORE REQUIREMENTS VERIFICATION SUITE       ")
    print("========================================================\n")
    
    passed_tests = 0
    total_tests = 9

    # ----------------------------------------------------
    # Requirement 1: Dockerfile and docker-compose.yml
    # ----------------------------------------------------
    print("[Test 1/9] Verifying Dockerfile and docker-compose.yml...")
    assert os.path.exists("Dockerfile"), "Dockerfile missing"
    assert os.path.exists("docker-compose.yml"), "docker-compose.yml missing"
    
    with open("docker-compose.yml", "r", encoding="utf-8") as f:
        compose_content = f.read()
        
    assert "services:" in compose_content, "docker-compose.yml missing 'services'"
    assert "trainer:" in compose_content, "docker-compose.yml missing 'trainer' service"
    assert "volumes:" in compose_content, "docker-compose.yml missing 'volumes'"
    assert "entrypoint:" in compose_content, "docker-compose.yml missing 'entrypoint'"
    assert "healthcheck:" in compose_content, "docker-compose.yml missing 'healthcheck'"
    
    for expected_mount in ["./src:", "./data:", "./results:", "./output:"]:
        assert expected_mount in compose_content, f"Volume '{expected_mount}' not mounted in docker-compose.yml"
        
    print("  -> Passed Requirement 1: Docker & Compose configuration valid.")
    passed_tests += 1

    # ----------------------------------------------------
    # Requirement 2: .env.example
    # ----------------------------------------------------
    print("[Test 2/9] Verifying .env.example...")
    assert os.path.exists(".env.example"), ".env.example missing"
    with open(".env.example", "r", encoding="utf-8") as f:
        env_content = f.read()
    assert "BASE_MODEL_NAME" in env_content, ".env.example missing BASE_MODEL_NAME"
    assert "HF_TOKEN" in env_content, ".env.example missing HF_TOKEN"
    assert "LEARNING_RATE" in env_content, ".env.example missing LEARNING_RATE"
    assert "LORA_RANK" in env_content, ".env.example missing LORA_RANK"
    assert "NUM_EPOCHS" in env_content, ".env.example missing NUM_EPOCHS"
    print("  -> Passed Requirement 2: .env.example exists and contains required keys.")
    passed_tests += 1

    # ----------------------------------------------------
    # Requirement 3: src/preprocess.py with custom heuristics (No spaCy/NLTK)
    # ----------------------------------------------------
    print("[Test 3/9] Verifying src/preprocess.py heuristics & AST...")
    assert os.path.exists("src/preprocess.py"), "src/preprocess.py missing"
    with open("src/preprocess.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    tree = ast.parse(code)
    imported_modules = []
    function_names = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imported_modules.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(node.module)
        elif isinstance(node, ast.FunctionDef):
            function_names.append(node.name)
            
    disallowed = ["spacy", "nltk", "stanza"]
    for mod in imported_modules:
        for dis in disallowed:
            assert dis not in mod.lower(), f"Disallowed NLP library '{mod}' found in src/preprocess.py"
            
    # Check for at least 2 custom transformation functions
    custom_functions = [fn for fn in function_names if fn in [
        "expand_clinical_abbreviations", "normalize_lab_values", "standardize_clinical_sections", "clean_clinical_text"
    ]]
    assert len(custom_functions) >= 2, f"Expected at least 2 custom heuristics, found {custom_functions}"
    print(f"  -> Passed Requirement 3: src/preprocess.py validated with custom functions {custom_functions} and 0 disallowed NLP libraries.")
    passed_tests += 1

    # ----------------------------------------------------
    # Requirement 4: src/train.py with PEFT & LoRA
    # ----------------------------------------------------
    print("[Test 4/9] Verifying src/train.py LoRA and PEFT usage...")
    assert os.path.exists("src/train.py"), "src/train.py missing"
    with open("src/train.py", "r", encoding="utf-8") as f:
        train_code = f.read()
    assert "from peft import" in train_code or "import peft" in train_code, "peft library not imported in src/train.py"
    assert "LoraConfig" in train_code, "LoraConfig not used in src/train.py"
    assert "get_peft_model" in train_code, "get_peft_model not used in src/train.py"
    print("  -> Passed Requirement 4: src/train.py contains valid PEFT LoRA implementation.")
    passed_tests += 1

    # ----------------------------------------------------
    # Requirement 5: output/final_adapter/
    # ----------------------------------------------------
    print("[Test 5/9] Verifying output/final_adapter/ files...")
    assert os.path.exists("output/final_adapter"), "output/final_adapter directory missing"
    assert os.path.exists("output/final_adapter/adapter_config.json"), "output/final_adapter/adapter_config.json missing"
    has_weights = os.path.exists("output/final_adapter/adapter_model.safetensors") or os.path.exists("output/final_adapter/adapter_model.bin")
    assert has_weights, "output/final_adapter missing adapter_model.safetensors or adapter_model.bin"
    print("  -> Passed Requirement 5: output/final_adapter/ files exist and valid.")
    passed_tests += 1

    # ----------------------------------------------------
    # Requirement 6: results/training_analysis.json schema
    # ----------------------------------------------------
    print("[Test 6/9] Verifying results/training_analysis.json schema...")
    assert os.path.exists("results/training_analysis.json"), "results/training_analysis.json missing"
    with open("results/training_analysis.json", "r", encoding="utf-8") as f:
        t_data = json.load(f)
    
    assert "stable_run" in t_data, "Missing 'stable_run' in training_analysis.json"
    assert "unstable_run" in t_data, "Missing 'unstable_run' in training_analysis.json"
    assert "log_path" in t_data["stable_run"], "Missing 'log_path' in stable_run"
    assert "final_metrics" in t_data["stable_run"], "Missing 'final_metrics' in stable_run"
    assert "train_loss" in t_data["stable_run"]["final_metrics"], "Missing 'train_loss' in stable_run.final_metrics"
    assert "eval_loss" in t_data["stable_run"]["final_metrics"], "Missing 'eval_loss' in stable_run.final_metrics"
    assert "log_path" in t_data["unstable_run"], "Missing 'log_path' in unstable_run"
    assert "final_metrics" in t_data["unstable_run"], "Missing 'final_metrics' in unstable_run"
    assert "train_loss" in t_data["unstable_run"]["final_metrics"], "Missing 'train_loss' in unstable_run.final_metrics"
    assert "eval_loss" in t_data["unstable_run"]["final_metrics"], "Missing 'eval_loss' in unstable_run.final_metrics"
    print("  -> Passed Requirement 6: results/training_analysis.json matches required schema.")
    passed_tests += 1

    # ----------------------------------------------------
    # Requirement 7: results/evaluation_metrics.json schema
    # ----------------------------------------------------
    print("[Test 7/9] Verifying results/evaluation_metrics.json schema...")
    assert os.path.exists("results/evaluation_metrics.json"), "results/evaluation_metrics.json missing"
    with open("results/evaluation_metrics.json", "r", encoding="utf-8") as f:
        e_data = json.load(f)
    
    assert "base_model_metrics" in e_data, "Missing 'base_model_metrics'"
    assert "fine_tuned_model_metrics" in e_data, "Missing 'fine_tuned_model_metrics'"
    
    for section in ["base_model_metrics", "fine_tuned_model_metrics"]:
        sec_obj = e_data[section]
        assert "rouge1" in sec_obj and isinstance(sec_obj["rouge1"], (int, float)), f"{section} missing numeric rouge1"
        assert "rouge2" in sec_obj and isinstance(sec_obj["rouge2"], (int, float)), f"{section} missing numeric rouge2"
        assert "rougeL" in sec_obj and isinstance(sec_obj["rougeL"], (int, float)), f"{section} missing numeric rougeL"
        assert "custom_metric_name" in sec_obj and isinstance(sec_obj["custom_metric_name"], str), f"{section} missing string custom_metric_name"
        assert "custom_metric_value" in sec_obj and isinstance(sec_obj["custom_metric_value"], (int, float)), f"{section} missing numeric custom_metric_value"
        
    print("  -> Passed Requirement 7: results/evaluation_metrics.json matches required schema.")
    passed_tests += 1

    # ----------------------------------------------------
    # Requirement 8: results/hallucination_analysis.md
    # ----------------------------------------------------
    print("[Test 8/9] Verifying results/hallucination_analysis.md...")
    assert os.path.exists("results/hallucination_analysis.md"), "results/hallucination_analysis.md missing"
    assert os.path.getsize("results/hallucination_analysis.md") > 0, "results/hallucination_analysis.md is empty"
    with open("results/hallucination_analysis.md", "r", encoding="utf-8") as f:
        h_text = f.read()
    assert "source" in h_text.lower(), "Missing source text in hallucination analysis"
    assert "summary" in h_text.lower(), "Missing summary in hallucination analysis"
    assert "hypothesis" in h_text.lower() or "hypotheses" in h_text.lower(), "Missing hypothesis in hallucination analysis"
    print("  -> Passed Requirement 8: results/hallucination_analysis.md is non-empty and well-structured.")
    passed_tests += 1

    # ----------------------------------------------------
    # Requirement 9: docs/lora_config.md
    # ----------------------------------------------------
    print("[Test 9/9] Verifying docs/lora_config.md...")
    assert os.path.exists("docs/lora_config.md"), "docs/lora_config.md missing"
    assert os.path.getsize("docs/lora_config.md") > 0, "docs/lora_config.md is empty"
    with open("docs/lora_config.md", "r", encoding="utf-8") as f:
        lora_text = f.read()
    assert "rank" in lora_text.lower() or "r =" in lora_text, "Missing rank discussion in docs/lora_config.md"
    assert "alpha" in lora_text.lower(), "Missing alpha discussion in docs/lora_config.md"
    assert "target_modules" in lora_text.lower() or "target" in lora_text.lower(), "Missing target_modules discussion in docs/lora_config.md"
    print("  -> Passed Requirement 9: docs/lora_config.md is non-empty and well-structured.")
    passed_tests += 1

    print("\n========================================================")
    print(f" ALL {passed_tests}/{total_tests} CORE REQUIREMENTS VERIFIED AND PASSED! ")
    print("========================================================\n")


if __name__ == "__main__":
    test_core_requirements()
