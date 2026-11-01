from agent.code_generator import build_codegen_prompt
from agent.design_profiles import build_design_guidance, select_design_profile


def test_task_app_selects_a_task_workflow_and_reuses_spec_context():
    spec = {
        "app_name": "Campus Tasks",
        "purpose": "Plan projects and deadlines",
        "entities": [{"name": "Task", "fields": [{"name": "priority"}, {"name": "due_date"}]}],
    }

    assert select_design_profile(spec) == "task_management"
    prompt = build_codegen_prompt(spec, "script.js", "Task page", user_prompt="Track tasks and due dates")
    assert "Domain profile: task management" in prompt
    assert "priority cues" in prompt
    assert "All data must come from the declared API" in prompt


def test_finance_app_gets_finance_layout_without_invented_features():
    spec = {
        "app_name": "Student Budget",
        "entities": [{"name": "Expenses", "fields": [{"name": "amount"}, {"name": "category"}]}],
    }

    guidance = build_design_guidance(spec, "Track monthly budgets and expenses")
    assert "Domain profile: personal finance" in guidance
    assert "transaction table or list" in guidance
    assert "Do not invent account balances" in guidance


def test_unrecognized_app_gets_generic_accessible_design_guidance():
    prompt = build_codegen_prompt({"app_name": "Neighborhood Garden"}, "style.css", "Styles")

    assert "Domain profile:" not in prompt
    assert "responsive layouts" in prompt
    assert "visible focus states" in prompt
