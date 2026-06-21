"""
Phase 5: Final System Specification & AI Adaptive Behavior Design
=================================================================
SCOPE: System behavior design layer. No model retraining.
INPUTS: Phase 3C severity hierarchy + Phase 4 design implication matrix
GUARDRAILS:
  - All rules trace back to TB-1 to TB-7 empirical severity structure
  - TB-4 (System Usability) is absolute override priority in all logic
  - Taxonomy TB-1 to TB-7 is frozen
  - No new inference or clustering
"""

import json
from pathlib import Path
from datetime import datetime

# Paths — anchored relative to this script's location (Topic-Modelling/)
BASE_DIR = Path(__file__).resolve().parent
PHASE3C_OUT = BASE_DIR / "phase3c_outputs"
PHASE4_OUT = BASE_DIR / "phase4_outputs"
OUT_DIR = BASE_DIR / "phase5_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Starting Phase 5: AI Adaptive System Logic Specification...")

# Load upstream severity data for traceability
with open(PHASE3C_OUT / "barrier_severity_ranking.json", encoding="utf-8") as f:
    severity_data = json.load(f)
severity_index = {r["barrier_category"]: r for r in severity_data["rankings"]}

# ─────────────────────────────────────────────────────────────────────────────
# TASK 5.1: TB-Driven AI Decision Rule Engine
# Maps each barrier to a rule set: trigger condition, response behavior,
# intensity level, escalation path. All rules trace back to severity scores.
# ─────────────────────────────────────────────────────────────────────────────
print("Task 5.1: Building AI Decision Rule Engine...")

decision_rules = {
    "artifact": "TB-Driven AI Decision Rule Engine",
    "version": "1.0",
    "generated_at": datetime.now().isoformat(),
    "grounding": "All rules derived from empirical severity scores (Phase 3C) — not assumptions",
    "override_hierarchy": [
        "LEVEL-0: TB-4 System Stability — ABSOLUTE OVERRIDE (supersedes all other rules)",
        "LEVEL-1: TB-7 Curriculum Alignment — STRUCTURAL PREREQUISITE",
        "LEVEL-2: TB-6 Access / Affordability — ENTRY GATE",
        "LEVEL-3: TB-1 Cognitive Adaptation — PERSONALIZATION LAYER",
        "LEVEL-4: TB-2 Engagement + TB-3 Support — EXPERIENCE OPTIMIZATION",
    ],
    "rules": {
        "TB-4": {
            "barrier": "System Usability Issues",
            "severity_score": severity_index["TB-4 System Usability Issues"]["total_severity"],
            "priority_tier": "P0 - CRITICAL / ABSOLUTE OVERRIDE",
            "trigger_conditions": [
                "App crash detected or session terminated unexpectedly",
                "Authentication failure after 1 retry",
                "Content load failure after 3s timeout",
                "Server error response (5xx)",
                "Offline state detected",
            ],
            "response_behaviors": [
                "Immediately cache session state to local storage",
                "Activate offline-first learning mode (cached content)",
                "Display graceful degradation UI (not blank screen or raw error)",
                "Suppress all engagement / gamification triggers during degraded state",
                "Log failure event with timestamp + context for post-mortem",
            ],
            "escalation_logic": "No escalation — TB-4 is the floor. Fix before any other adaptive logic runs.",
            "ai_action": "AI Tutor SUSPENDED during TB-4 state — system recovery takes precedence",
        },
        "TB-7": {
            "barrier": "Content Quality Mismatch",
            "severity_score": severity_index["TB-7 Content Quality Mismatch"]["total_severity"],
            "priority_tier": "P0 - CRITICAL / STRUCTURAL PREREQUISITE",
            "trigger_conditions": [
                "Onboarding diagnostic score indicates curriculum level mismatch (>2 grade levels)",
                "User spends >3 sessions failing same prerequisite skill",
                "User explicitly selects curriculum type that doesn't match enrolled grade",
                "Content completion rate drops below 30% in a topic",
            ],
            "response_behaviors": [
                "Trigger curriculum realignment: re-run placement diagnostic",
                "Activate prerequisite knowledge graph traversal — serve upstream concepts first",
                "Flag content as 'curriculum_mismatch_detected' in session metadata",
                "Recommend alternative learning path from prerequisite graph",
            ],
            "escalation_logic": [
                "LEVEL-1: Suggest prerequisite content (soft nudge)",
                "LEVEL-2: Auto-reroute to prerequisite path (medium intervention)",
                "LEVEL-3: Lock advancement until prerequisite mastery threshold met (strong gate)",
            ],
            "ai_action": "AI Tutor engages with: 'Sepertinya ada konsep dasar yang perlu kita perkuat dulu sebelum lanjut.'",
        },
        "TB-6": {
            "barrier": "Cost / Affordability Barrier",
            "severity_score": severity_index["TB-6 Cost / Affordability Barrier"]["total_severity"],
            "priority_tier": "P1 - HIGH",
            "trigger_conditions": [
                "User hits paywall on core curriculum content",
                "User session engagement drops after encountering premium gate",
                "User has completed all available free content",
            ],
            "response_behaviors": [
                "Surface effort-based content unlock: 'Selesaikan 5 latihan untuk membuka materi ini'",
                "Display freemium pathway: highlight which content is permanently free",
                "Flag user as 'access_constrained' for subsidy pathway eligibility check",
                "Never show premium upsell during active struggle/failure states (bad timing)",
            ],
            "escalation_logic": [
                "LEVEL-1: Show freemium alternatives inline",
                "LEVEL-2: Offer effort-based unlock challenge",
                "LEVEL-3: Trigger subsidy/scholarship pathway if flagged eligible",
            ],
            "ai_action": "AI Tutor: 'Kamu bisa akses materi ini secara gratis dengan menyelesaikan latihan berikut.'",
        },
        "TB-1": {
            "barrier": "Cognitive Difficulty",
            "severity_score": severity_index["TB-1 Cognitive Difficulty"]["total_severity"],
            "priority_tier": "P1 - HIGH (minority at-risk targeting)",
            "important_note": "3.3% negative ratio — do NOT over-intervene. Majority succeeds. Target at-risk signals specifically.",
            "trigger_conditions": [
                "User answers <50% correct on 3+ consecutive exercises in same skill",
                "Time-on-task exceeds 3x average for similar learner profile",
                "User explicitly requests explanation: taps help/explain button",
                "User abandons content mid-session after encountering difficulty spike",
            ],
            "response_behaviors": [
                "Switch to multi-modal explanation: offer animation/visual if text was shown first",
                "Simplify vocabulary in AI tutor response (reduce Flesch reading complexity)",
                "Break concept into smaller micro-steps (ZPD chunking)",
                "Do NOT re-show identical failed content immediately — use spaced variant",
            ],
            "escalation_logic": [
                "LEVEL-1: Provide alternate explanation format (soft)",
                "LEVEL-2: Engage AI Tutor for real-time Q&A (medium)",
                "LEVEL-3: Reroute to prerequisite concept (strong — overlaps TB-7 logic)",
            ],
            "ai_action": "AI Tutor: 'Coba kita lihat dari sudut pandang berbeda. Bayangkan seperti ini...' + visual metaphor",
        },
        "TB-2": {
            "barrier": "Engagement & Motivation Problem",
            "severity_score": severity_index["TB-2 Engagement & Motivation Problem"]["total_severity"],
            "priority_tier": "P2 - MEDIUM (downstream optimization)",
            "important_note": "9.5% negative ratio — engagement is an optimization, not a blocker. NEVER prioritize over P0 layers.",
            "trigger_conditions": [
                "Session frequency drops below user's 7-day average",
                "User skips optional practice 3+ times consecutively",
                "User closes app within 60 seconds of opening (bounce)",
            ],
            "response_behaviors": [
                "Trigger streak preservation notification (not generic push)",
                "Surface short-form micro-content (<3 min) to reduce friction",
                "Celebrate recent progress: 'Kamu sudah belajar X menit minggu ini!'",
                "Do NOT trigger engagement nudges during TB-4 or TB-7 active states",
            ],
            "escalation_logic": [
                "LEVEL-1: In-app progress celebration (passive)",
                "LEVEL-2: Personalized challenge recommendation (active)",
                "LEVEL-3: Peer comparison / leaderboard (social proof)",
            ],
            "ai_action": "AI Tutor: 'Kamu hampir selesai bab ini! Yuk selesaikan dalam 5 menit lagi.'",
        },
        "TB-3": {
            "barrier": "Lack of Learning Support",
            "severity_score": severity_index["TB-3 Lack of Learning Support"]["total_severity"],
            "priority_tier": "P2 - MEDIUM (downstream optimization)",
            "trigger_conditions": [
                "User explicitly searches for a topic not finding results",
                "User fails exercise and taps 'Lihat Pembahasan' (view explanation)",
                "User attempts same question >2 times incorrectly",
            ],
            "response_behaviors": [
                "Activate AI Tutor context-aware Q&A session",
                "Surface related question bank items ranked by relevance",
                "Offer step-by-step worked example for failed question",
            ],
            "escalation_logic": [
                "LEVEL-1: Show worked example (passive support)",
                "LEVEL-2: AI Tutor inline hints (active support)",
                "LEVEL-3: Full AI Tutor dialogue session (deep support)",
            ],
            "ai_action": "AI Tutor: 'Boleh aku bantu? Coba kita kerjakan langkah demi langkah.'",
        },
    },
}

with open(OUT_DIR / "adaptive_decision_rule_engine.json", "w", encoding="utf-8") as f:
    json.dump(decision_rules, f, indent=4, ensure_ascii=False)
print("  -> adaptive_decision_rule_engine.json saved.")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 5.2: Learning Flow State Machine
# ─────────────────────────────────────────────────────────────────────────────
print("Task 5.2: Designing Learning Flow State Machine...")

state_machine = {
    "artifact": "Learning Flow State Machine",
    "model_type": "Finite State Machine (FSM) — logical/behavioral specification",
    "important_note": "This is a behavioral design specification, not runtime code. Embed TB-7 curriculum checkpoints and TB-4 recovery states.",
    "states": {
        "ONBOARDING": {
            "description": "User enters system for first time",
            "entry_actions": [
                "Run curriculum placement diagnostic (TB-7 prerequisite)",
                "Check system stability (TB-4 pre-check)",
                "Set access tier (TB-6 equity gate)",
            ],
            "transitions": {
                "diagnostic_passed": "ACTIVE_LEARNING",
                "mismatch_detected": "CURRICULUM_REALIGNMENT",
                "system_failure": "DEGRADED_MODE",
                "access_blocked": "ACCESS_LIMITATION_FLOW",
            },
        },
        "ACTIVE_LEARNING": {
            "description": "User is progressing through assigned content correctly",
            "entry_actions": [
                "Load curriculum-aligned content (TB-7 validated path)",
                "Activate engagement loop (TB-2 low-priority)",
            ],
            "transitions": {
                "cognitive_struggle_detected": "COGNITIVE_SUPPORT_MODE",
                "content_mismatch_detected": "CURRICULUM_REALIGNMENT",
                "system_failure": "DEGRADED_MODE",
                "engagement_drop": "MOTIVATION_RECOVERY",
                "support_request": "AI_TUTOR_SESSION",
                "content_completed": "PROGRESS_CHECKPOINT",
            },
        },
        "COGNITIVE_SUPPORT_MODE": {
            "description": "Learner is struggling with specific content (TB-1 trigger)",
            "entry_actions": [
                "Activate alternate explanation format",
                "Reduce task complexity (ZPD chunking)",
                "Surface AI Tutor hint (TB-3 support)",
            ],
            "transitions": {
                "struggle_resolved": "ACTIVE_LEARNING",
                "struggle_deepens": "AI_TUTOR_SESSION",
                "prerequisite_gap_found": "CURRICULUM_REALIGNMENT",
                "system_failure": "DEGRADED_MODE",
            },
        },
        "CURRICULUM_REALIGNMENT": {
            "description": "Curriculum mismatch detected — rerouting to prerequisite path (TB-7)",
            "entry_actions": [
                "Run prerequisite graph traversal",
                "Serve upstream foundational content",
                "Notify user with educational framing (not error message)",
            ],
            "transitions": {
                "prerequisites_met": "ACTIVE_LEARNING",
                "system_failure": "DEGRADED_MODE",
            },
        },
        "AI_TUTOR_SESSION": {
            "description": "Deep AI Tutor dialogue for TB-1/TB-3 support needs",
            "entry_actions": [
                "Load conversation context (last 3 failed attempts)",
                "Activate scaffolding dialogue policy",
                "Respect TB-4 state — suspend if system is degraded",
            ],
            "transitions": {
                "issue_resolved": "ACTIVE_LEARNING",
                "user_exits": "ACTIVE_LEARNING",
                "system_failure": "DEGRADED_MODE",
            },
        },
        "MOTIVATION_RECOVERY": {
            "description": "Engagement drop detected (TB-2 trigger)",
            "entry_actions": [
                "Surface short micro-content or progress celebration",
                "Do NOT trigger if TB-4 or TB-7 states are active",
            ],
            "transitions": {
                "engagement_restored": "ACTIVE_LEARNING",
                "user_exits": "SESSION_END",
            },
        },
        "ACCESS_LIMITATION_FLOW": {
            "description": "User hit affordability/paywall barrier (TB-6)",
            "entry_actions": [
                "Surface freemium content alternatives",
                "Offer effort-based unlock challenge",
                "Check subsidy eligibility flag",
            ],
            "transitions": {
                "access_granted": "ACTIVE_LEARNING",
                "user_exits": "SESSION_END",
            },
        },
        "DEGRADED_MODE": {
            "description": "System failure state (TB-4 override — highest priority)",
            "entry_actions": [
                "Cache current session state to local storage immediately",
                "Activate offline content bundle if available",
                "Suppress ALL non-critical features (engagement, gamification, AI Tutor)",
                "Display graceful degradation UI with honest status message",
            ],
            "transitions": {
                "system_restored": "ACTIVE_LEARNING",
                "offline_content_available": "OFFLINE_LEARNING",
                "no_content_available": "SESSION_END",
            },
        },
        "OFFLINE_LEARNING": {
            "description": "Minimal continuity mode — offline-first content only (TB-4 resilience)",
            "entry_actions": [
                "Load cached curriculum content only",
                "Disable AI Tutor, server-dependent features",
                "Enable local-only progress tracking",
            ],
            "transitions": {
                "connectivity_restored": "ACTIVE_LEARNING",
                "user_exits": "SESSION_END",
            },
        },
        "PROGRESS_CHECKPOINT": {
            "description": "User completed a content unit — validate advancement",
            "entry_actions": [
                "Check TB-7 curriculum alignment for next unit",
                "Update learner knowledge state in knowledge tracing model",
                "Trigger TB-2 engagement celebration (low priority)",
            ],
            "transitions": {
                "next_unit_aligned": "ACTIVE_LEARNING",
                "mismatch_in_next_unit": "CURRICULUM_REALIGNMENT",
            },
        },
        "SESSION_END": {
            "description": "Session terminated (voluntary or system-forced)",
            "entry_actions": [
                "Persist session state and progress",
                "Schedule re-engagement notification (TB-2 — delayed, respectful timing)",
            ],
            "transitions": {},
        },
    },
}

with open(OUT_DIR / "learning_flow_state_machine.json", "w", encoding="utf-8") as f:
    json.dump(state_machine, f, indent=4, ensure_ascii=False)
print("  -> learning_flow_state_machine.json saved.")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 5.3: AI Tutor Behavioral Policy
# ─────────────────────────────────────────────────────────────────────────────
print("Task 5.3: Defining AI Tutor Behavioral Policy...")

ai_tutor_policy = {
    "artifact": "AI Tutor Behavioral Policy for Adaptive Learning",
    "grounding": "TB-1 (Cognitive) + TB-3 (Support) severity signals, with TB-4 override awareness",
    "core_principles": [
        "PRINCIPLE-1: AI Tutor does not activate during TB-4 degraded system states",
        "PRINCIPLE-2: AI Tutor addresses cognitive and support needs — it does not fix system failures",
        "PRINCIPLE-3: Over-intervention is harmful — AI should step back once struggle is resolved",
        "PRINCIPLE-4: All AI responses must use Indonesian (Bahasa Indonesia) appropriate to learner level",
        "PRINCIPLE-5: AI Tutor is scaffolding, not answer-giving — Socratic dialogue preferred",
    ],
    "behavioral_policies": {
        "POLICY-TB1-COGNITIVE": {
            "trigger_barrier": "TB-1 Cognitive Difficulty",
            "activation_threshold": "User fails 50% of exercises in same skill OR explicitly requests help",
            "response_strategy": "Scaffolded Socratic dialogue using ZPD-aligned prompts",
            "dialogue_rules": [
                "Start with a diagnostic question: 'Bagian mana yang terasa paling membingungkan?'",
                "Use concrete analogies (local cultural context when possible)",
                "Break explanation into max 3 steps",
                "Confirm understanding before advancing: 'Apakah penjelasan ini sudah lebih jelas?'",
                "If 3 attempts fail, escalate to prerequisite reroute (TB-7 handoff)",
            ],
            "forbidden_behaviors": [
                "Do NOT give direct answer to exercise",
                "Do NOT repeat identical explanation twice",
                "Do NOT remain active after struggle is resolved",
            ],
        },
        "POLICY-TB3-SUPPORT": {
            "trigger_barrier": "TB-3 Lack of Learning Support",
            "activation_threshold": "User explicitly searches for content OR taps help after failure",
            "response_strategy": "Resource surfacing + guided practice",
            "dialogue_rules": [
                "Acknowledge the specific request: 'Oke, aku bantu cari materi tentang X.'",
                "Surface most relevant question bank items (top 3)",
                "Offer worked example if user fails second time",
                "Conclude with a practice question to verify understanding",
            ],
            "forbidden_behaviors": [
                "Do NOT provide passive responses without actionable next step",
                "Do NOT overwhelm with >3 resources at once",
            ],
        },
        "POLICY-SYSTEM-AWARENESS": {
            "purpose": "Ensures AI Tutor respects TB-4 override",
            "rules": [
                "Before any AI Tutor activation: check system health state",
                "If state == DEGRADED_MODE or OFFLINE_LEARNING: SUSPEND AI Tutor",
                "If state == CURRICULUM_REALIGNMENT: coordinate with prerequisite graph before responding",
                "AI Tutor must never contradict or override curriculum path set by TB-7 logic",
            ],
        },
        "POLICY-NON-INTERVENTION": {
            "purpose": "Prevent over-intervention in non-critical cases (TB-1 majority is positive)",
            "rules": [
                "If user negative_ratio for session < 10%: do NOT proactively activate AI Tutor",
                "If user is in first session: observe only, do not intervene unless explicit request",
                "After 3 consecutive correct answers: deactivate active support mode",
            ],
        },
    },
}

with open(OUT_DIR / "ai_tutor_behavioral_policy.json", "w", encoding="utf-8") as f:
    json.dump(ai_tutor_policy, f, indent=4, ensure_ascii=False)
print("  -> ai_tutor_behavioral_policy.json saved.")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 5.4: System Resilience & Failure Recovery Protocol
# ─────────────────────────────────────────────────────────────────────────────
print("Task 5.4: Defining System Resilience Protocol...")

resilience_protocol = {
    "artifact": "System Resilience & Failure Recovery Protocol",
    "grounding": "TB-4 (severity: 1687.12, 49.8% neg ratio) — highest empirical severity in corpus",
    "design_philosophy": "Stability over feature richness. Minimum viable learning continuity must always be preserved.",
    "failure_tiers": {
        "TIER-1 TRANSIENT": {
            "description": "Temporary failures (network hiccup, slow load)",
            "threshold": "Content load > 3s OR single request failure",
            "response": [
                "Show loading state with honest wait time estimate",
                "Auto-retry once silently",
                "If retry fails → escalate to TIER-2",
            ],
        },
        "TIER-2 DEGRADED": {
            "description": "Partial system failure (auth issues, content unavailable, server error)",
            "threshold": "2+ consecutive failures OR authentication failure",
            "response": [
                "Cache session state immediately",
                "Switch to offline content bundle if available",
                "Suppress AI Tutor and gamification features",
                "Show status: 'Koneksi terbatas — kami menyimpan progresmu secara otomatis'",
            ],
        },
        "TIER-3 CRITICAL": {
            "description": "Full system outage (app crash, complete offline, auth system down)",
            "threshold": "App crash OR complete connectivity loss",
            "response": [
                "Enter OFFLINE_LEARNING state (cached curriculum only)",
                "Disable all server-dependent features",
                "Show minimal UI: progress, current topic, offline exercises only",
                "Queue all progress events for sync on reconnect",
                "On reconnect: sync → validate curriculum alignment → resume from last checkpoint",
            ],
        },
    },
    "offline_first_requirements": [
        "Core curriculum content for enrolled courses must be pre-cached at onboarding",
        "Last 3 sessions of exercises must be available offline",
        "Progress events must be stored locally and synced asynchronously",
        "Offline mode UI must be indistinguishable from online for cached content",
    ],
    "recovery_sequence": [
        "Step 1: Detect restoration of connectivity",
        "Step 2: Sync queued progress events to server",
        "Step 3: Validate curriculum alignment (TB-7 checkpoint)",
        "Step 4: Restore full feature set progressively (not all at once)",
        "Step 5: Resume from last stable learning state",
        "Step 6: Log failure event + recovery time for system health monitoring",
    ],
}

with open(OUT_DIR / "system_resilience_protocol.json", "w", encoding="utf-8") as f:
    json.dump(resilience_protocol, f, indent=4, ensure_ascii=False)
print("  -> system_resilience_protocol.json saved.")

# ─────────────────────────────────────────────────────────────────────────────
# MASTER OUTPUT: SR System Logic Specification (unified reference document)
# ─────────────────────────────────────────────────────────────────────────────
print("Assembling Master SR System Logic Specification...")

master_spec = {
    "artifact": "SR Adaptive Learning System Logic Specification",
    "version": "1.0-FINAL",
    "generated_at": datetime.now().isoformat(),
    "pipeline_provenance": "LDA_k8 (Phase 3A) → TB Taxonomy (3B) → Severity Model (3C) → Blueprint Prioritization (3D) → Design Synthesis (4) → System Specification (5)",
    "taxonomy_version": "TB-1 to TB-7 (frozen)",
    "empirical_grounding": "All behavior rules derived from 41,797 Indonesian educational platform reviews",
    "override_hierarchy": decision_rules["override_hierarchy"],
    "implementation_sequencing": [
        "PHASE-A: Build TB-4 resilience layer (offline-first, crash recovery, graceful degradation)",
        "PHASE-B: Build TB-7 curriculum engine (prerequisite graph, diagnostic placement, path sequencing)",
        "PHASE-C: Implement TB-6 access model (freemium, effort-based unlocks, equity pathways)",
        "PHASE-D: Deploy TB-1 cognitive adaptation (multi-modal content, ZPD chunking, difficulty calibration)",
        "PHASE-E: Activate AI Tutor (TB-3 support + TB-1 scaffolding) — only after Phases A-D stable",
        "PHASE-F: Engage TB-2 optimization layer (gamification, streaks, social features) — last",
    ],
    "component_references": {
        "adaptive_decision_rules": "phase5_outputs/adaptive_decision_rule_engine.json",
        "state_machine": "phase5_outputs/learning_flow_state_machine.json",
        "ai_tutor_policy": "phase5_outputs/ai_tutor_behavioral_policy.json",
        "resilience_protocol": "phase5_outputs/system_resilience_protocol.json",
    },
    "critical_design_constraints": [
        "TB-4 system failure OVERRIDES all other adaptive logic — no exceptions",
        "AI Tutor MUST NOT activate during degraded/offline system states",
        "Curriculum alignment (TB-7) MUST precede personalization features",
        "Engagement features (TB-2) MUST NOT be prioritized over structural barriers (TB-4, TB-7)",
        "All behavior rules must remain traceable to empirical severity hierarchy",
    ],
}

with open(OUT_DIR / "sr_system_logic_specification.json", "w", encoding="utf-8") as f:
    json.dump(master_spec, f, indent=4, ensure_ascii=False)
print("  -> sr_system_logic_specification.json saved.")

print("\nPhase 5 completed! All outputs saved to phase5_outputs/")
print("Pipeline complete: Phase 3A -> Phase 5 fully executed.")
