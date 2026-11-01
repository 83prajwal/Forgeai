"""Small deterministic design briefs for domain-aware generated applications.

These profiles enrich the existing per-file prompts locally; they do not add
provider calls or prescribe features that are absent from the application spec.
"""
from __future__ import annotations

import re
from typing import Any


_PROFILES: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    (
        "task_management",
        ("task", "tasks", "todo", "project", "milestone", "deadline", "kanban"),
        "Use a focused work-management layout: a compact progress overview, clear status and priority cues, and a task list or board that makes due dates scannable. Keep the primary task action easy to find.",
        "Represent status, priority, and dates only when present in the spec. Search, filtering, create, edit, and delete controls must call the declared API and show loading, empty, success, and error states.",
    ),
    (
        "personal_finance",
        ("finance", "budget", "expense", "expenses", "spending", "transaction", "transactions", "income", "money"),
        "Use a calm, numbers-first finance layout: readable totals, a transaction table or list with aligned amounts and dates, and category or budget summaries when those entities exist. Use restrained color for positive and negative values, with currency formatting appropriate to the spec.",
        "Use only the financial records, budgets, and calculations defined by the spec. Do not invent account balances, bank connections, income, or transactions. Forms must validate numeric amounts and use declared API routes.",
    ),
    (
        "inventory",
        ("inventory", "stock", "product", "products", "warehouse", "sku", "supplier"),
        "Use an operations layout with a dense but readable product table, prominent quantities, and easy scanning by SKU, category, and stock state.",
        "Show low-stock warnings only when a threshold or rule exists in the spec. Product and supplier actions must use the declared API and persist to SQLite.",
    ),
    (
        "library_management",
        ("library", "book", "books", "catalog", "borrow", "borrowed", "lending"),
        "Use a catalog-first layout with clear book titles, authors, availability, and compact search/filter controls. Keep issue and return actions visible only when requested and supported by the data model.",
        "Do not fabricate books or borrowers. Availability and lending actions must be backed by fields and API routes in the spec.",
    ),
    (
        "crm",
        ("crm", "customer", "customers", "client", "clients", "sales pipeline", "lead", "leads"),
        "Use a relationship-focused layout with contact details, clear pipeline stages when present, and a readable activity timeline only when the spec includes activity data.",
        "Do not add outreach, email sending, or activity history unless the spec defines those features and their data/API support.",
    ),
    (
        "ecommerce",
        ("ecommerce", "e-commerce", "online store", "shop", "shopping", "cart", "checkout", "product catalog"),
        "Use a product-first responsive grid with clear product details and an obvious cart summary when those workflows are requested.",
        "Do not imply payment processing, shipping, or checkout is real unless the spec explicitly supports it. Implement only interactions backed by available data and routes.",
    ),
    (
        "fitness",
        ("fitness", "workout", "exercise", "training", "gym", "health log"),
        "Use an energetic but calm progress layout with sessions, dates, durations, and goals easy to scan. Charts should be derived from saved records and have an empty state.",
        "Do not invent health measurements or progress. Render summaries from the app's actual stored records.",
    ),
    (
        "attendance",
        ("attendance", "student attendance", "class roster", "class register"),
        "Use a roster-first education layout with class/date context and unmistakable present, late, absent, or excused status treatment when those values exist.",
        "Attendance changes must be saved through the declared API. Do not prepopulate students or attendance records.",
    ),
    (
        "resume_and_career",
        ("resume", "cv", "career profile", "work experience", "portfolio"),
        "Use a polished editorial profile layout with clear hierarchy for contact details, experience, education, and skills when those sections are in the spec.",
        "Keep all profile sections editable and persisted. Do not invent a person's identity, employment history, or credentials.",
    ),
)


def select_design_profile(spec: dict[str, Any] | None, user_prompt: str = "") -> str:
    """Select a profile from the request and structured spec using word matches."""
    spec = spec if isinstance(spec, dict) else {}
    parts = [user_prompt, str(spec.get("app_name", "")), str(spec.get("purpose", ""))]
    for entity in spec.get("entities", []) if isinstance(spec.get("entities"), list) else []:
        if isinstance(entity, dict):
            parts.extend((str(entity.get("name", "")), " ".join(
                str(field.get("name", "")) for field in entity.get("fields", []) if isinstance(field, dict)
            )))
    for page in spec.get("pages", []) if isinstance(spec.get("pages"), list) else []:
        if isinstance(page, dict):
            parts.extend((str(page.get("name", "")), str(page.get("purpose", ""))))
    text = " ".join(parts).casefold()

    best_name = "general_product"
    best_score = 0
    for name, keywords, _layout, _requirements in _PROFILES:
        score = sum(
            1 for keyword in keywords
            if re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text)
        )
        if score > best_score:
            best_name, best_score = name, score
    return best_name


def build_design_guidance(spec: dict[str, Any] | None, user_prompt: str = "") -> str:
    """Return concise style and implementation direction for the chosen domain."""
    selected = select_design_profile(spec, user_prompt)
    profile = next((item for item in _PROFILES if item[0] == selected), None)
    if profile is None:
        return (
            "Select a layout that fits the app's stated purpose and entities. Use the spec's accent color, "
            "clear typography and spacing, responsive layouts, visible focus states, and real API-backed "
            "empty/loading/success/error states. Do not add features or demonstration records that the spec does not request."
        )
    _name, _keywords, layout, requirements = profile
    return (
        f"Domain profile: {selected.replace('_', ' ')}.\n"
        f"Layout direction: {layout}\n"
        f"Functional constraints: {requirements}\n"
        "Use the accent color and theme tone from the spec. Keep typography, spacing, borders, and control states "
        "consistent; adapt the layout at tablet and mobile widths; provide keyboard-visible focus and labeled controls. "
        "All data must come from the declared API. Never use hardcoded sample records to make a dashboard look populated."
    )
