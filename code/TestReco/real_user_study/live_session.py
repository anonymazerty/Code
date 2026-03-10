import os
import json
import time
import importlib
import inspect
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

from envs.education_env import EducationEnv
from generators.factory import model_factory

# Pretest estimation helpers (unchanged interface)
from real_user_study.initial_estimation import estimate_theta_rasch, theta_to_mastery

# Read orchestrator type from settings.py (single source of truth)
try:
    from real_user_study.settings import ORCHESTRATOR_TYPE as SETTINGS_ORCHESTRATOR_TYPE
except Exception:
    SETTINGS_ORCHESTRATOR_TYPE = "tool_call"


@dataclass
class StepResult:
    selected_qids: List[int]
    rewards: Dict[str, float]
    mastery_before: float
    mastery_after: float
    rolling_accuracy: float
    info: Dict[str, Any]


class RealUserEngine:
    """
    Human-in-the-loop engine.

    Key requirement for reusing existing trained policies:
      - The observation/state dimension must match the policies.
      - Your existing policies were trained with num_difficulties=5 => state_dim=11 (for 1 skill).
      - Fundamental Math has 6 original difficulty levels -> we remap 6 -> 5 FOR THE ENV ONLY.
      - UI can still display 1..6 (kept in self.difficulties_display).
    """

    def __init__(
        self,
        topic: str,
        questions: List[dict],
        difficulties: Dict[int, dict],
        max_steps: int,
        questions_per_step: int,
        objectives: List[str],
        seed: int = 42,
        ncc_window: int = 2,
    ):
        self.topic = topic
        self.questions = questions

        # Keep the original difficulties for UI display (Level 1..6)
        self.difficulties_display: Dict[int, dict] = {int(k): v for k, v in difficulties.items()}

        # Remap to 5 levels FOR THE ENV (so state_dim matches trained policies)
        self.difficulties_env: Dict[int, dict] = self._remap_difficulties_to_5_bins(self.difficulties_display)

        self.objectives = objectives
        self.max_steps = max_steps
        self.questions_per_step = questions_per_step
        self.seed = seed
        self.ncc_window = ncc_window

        # Create env with 1 skill and 5 difficulties
        self.env = EducationEnv(
            skills=[topic],
            num_questions=len(questions),
            objectives=objectives,
            response_model=None,                 # humans answer
            target_skill_bundle=[topic],
            max_steps=max_steps,
            early_stop_threshold=0.8,
            seed=seed,
            ncc_window=ncc_window,
        )

        # Set difficulty levels using the remapped dict (5 bins)
        self.env.set_difficulty_levels(self.difficulties_env)

        # Single-skill mapping
        for qid in range(len(questions)):
            self.env.set_question_skills_difficulty(qid, [topic])

        self.env.questions_per_step = questions_per_step

        self._manual_reset_state()

        # Make this generic (not ToolCallOrchestrator-specific)
        self.orchestrator: Optional[Any] = None

        print("[RealUserEngine] Initialized.")
        print(f"[RealUserEngine] Topic(UI): {self.topic}")
        print(f"[RealUserEngine] Env difficulty bins forced to: {self.env.difficulty_levels}")
        print(f"[RealUserEngine] Env state_dim (computed by env): {getattr(self.env, 'state_dim', 'unknown')}")

    # ---------------------------
    # Difficulty remapping
    # ---------------------------

    def _remap_difficulties_to_5_bins(self, diffs_display: Dict[int, dict]) -> Dict[int, dict]:
        """
        Fundamental Math has 6 original levels.
        Policies were trained with 5 levels => we remap original 1..6 -> env 1..5:
          1->1, 2->2, 3->3, 4->4, 5->5, 6->5 (merge hardest bin)
        Env scaled difficulty becomes: env_level / 5
        """
        mapped: Dict[int, dict] = {}
        counts = {i: 0 for i in range(1, 7)}
        counts_m = {i: 0 for i in range(1, 6)}

        for qid, d in diffs_display.items():
            try:
                od = float(d.get("original_difficulty", 0.0))
            except Exception:
                od = 0.0

            # UI/original difficulty (1..6)
            ui_lvl = int(round(od)) if od > 0 else 1
            ui_lvl = max(1, min(6, ui_lvl))
            counts[ui_lvl] += 1

            # Env difficulty (1..5)
            env_lvl = min(ui_lvl, 5)  # merges 6->5
            counts_m[env_lvl] += 1

            # Scaled in [1/5, 1.0]
            env_scaled = float(env_lvl) / 5.0

            mapped[int(qid)] = {
                "original_difficulty": float(env_lvl),     # env sees 1..5
                "scaled_difficulty": env_scaled,           # env sees 1/5..1
                # keep UI fields for debugging (not used by env)
                "original_difficulty_ui": float(ui_lvl),
                "scaled_difficulty_ui": float(d.get("scaled_difficulty", env_scaled)),
            }

        print("\n[RealUserEngine] ===== Difficulty remap (UI 1..6 -> ENV 1..5) =====")
        print(f"[RealUserEngine] UI counts per level: {counts}")
        print(f"[RealUserEngine] ENV counts per level: {counts_m}")
        print("[RealUserEngine] ==================================================\n")

        return mapped

    # ---------------------------
    # Manual reset
    # ---------------------------

    def _manual_reset_state(self) -> None:
        self.env.mastery = 0.4 * np.ones(self.env.num_skills, dtype=np.float32)
        self.env.current_step = 0

        self.env.all_failed_questions = {}
        self.env.cleared_questions = {}
        self.env.seen_materials = []

        self.env.ncc_tracking = {i: {} for i in range(self.env.num_skills)}
        self.env.aptitude_cache = {}
        self.env.experience_cache = {}
        self.env.gap_cache = {}

        self.env.failed_questions_ratio = np.zeros(
            (self.env.num_skills, self.env.difficulty_levels), dtype=np.float32
        )
        self.env.skill_difficulty_accuracy = np.zeros(
            (self.env.num_skills, self.env.difficulty_levels), dtype=np.float32
        )
        self.env.skill_difficulty_counts = np.zeros(
            (self.env.num_skills, self.env.difficulty_levels), dtype=np.uint8
        )

        if hasattr(self.env, "_update_aptitude_cache"):
            self.env._update_aptitude_cache()
        if hasattr(self.env, "_update_experience_cache"):
            self.env._update_experience_cache()
        if hasattr(self.env, "_update_gap_cache"):
            self.env._update_gap_cache()

    # ---------------------------
    # Pretest (Qualification test)
    # ---------------------------

    def apply_pretest(self, qids: List[int], correctness: List[bool]) -> float:
        """
        IMPORTANT: This method does NOT select pretest questions.
        It only applies the submitted (possibly filtered) pretest answers.

        Your new logic (guessing detection, filtering, 8/12 gating) happens in app.py.
        Here we simply:
          - estimate mastery using ENV-scaled difficulties (5-bin remap)
          - update env state / caches consistently
        """
        if len(qids) != len(correctness):
            raise ValueError("apply_pretest: qids and correctness must have same length")

        # Use ENV-scaled difficulties (remapped to 5 bins)
        scaled = [float(self.env.question_skills_difficulty_map[q]["scaled_difficulty"]) for q in qids]
        theta0 = estimate_theta_rasch(scaled, correctness)
        mastery0 = float(theta_to_mastery(theta0))

        self.env.mastery[0] = mastery0

        print("\n[Pretest] ===== Submitting pretest (filtered qids if any) =====")
        print(f"[Pretest] mastery0={mastery0:.3f}, theta0={theta0:.3f}")
        for qid, correct in zip(qids, correctness):
            # Debug prints, robust to float UI levels
            ui_lvl_raw = self.difficulties_display.get(qid, {}).get("original_difficulty", 1)
            try:
                ui_lvl = int(round(float(ui_lvl_raw)))
            except Exception:
                ui_lvl = 1

            env_lvl = int(self.difficulties_env[qid].get("original_difficulty", 1))
            env_sd = float(self.difficulties_env[qid].get("scaled_difficulty", 0.2))
            print(
                f"[Pretest] qid={qid} "
                f"UI_level={ui_lvl} ENV_level={env_lvl} ENV_scaled={env_sd:.2f} "
                f"correct={bool(correct)}"
            )
        print("[Pretest] =====================================================\n")

        for qid, correct in zip(qids, correctness):
            qinfo = self.env.question_skills_difficulty_map[qid]
            skills = np.array([0], dtype=int)

            self.env.seen_materials.append((qid, qinfo["scaled_difficulty"], bool(correct)))

            self.env._update_skill_difficulty_accuracy(
                qid, bool(correct), skills, qinfo["original_difficulty"]
            )

            self.env._update_failed_question(
                qid, skills, bool(correct),
                qinfo["original_difficulty"],
                qinfo["scaled_difficulty"],
            )

        self.env._update_aptitude_cache()
        self.env._update_experience_cache()
        self.env._update_gap_cache()

        return mastery0

    # ---------------------------
    # Policies discovery + filtering
    # ---------------------------

    def _discover_policy_folders(self, policy_folders: List[str]) -> List[str]:
        """
        Accepts:
          - list of policy folders (each contains config.json)
          - OR a single root folder that contains many policy subfolders (e.g. ["results/"])
        Returns: list of actual policy folders that contain config.json
        """
        if not policy_folders:
            raise ValueError("No policy folders provided.")

        # if single root folder, search inside it
        if len(policy_folders) == 1 and os.path.isdir(policy_folders[0]):
            root = policy_folders[0]
            found = []
            for dirpath, dirnames, filenames in os.walk(root):
                if "config.json" in filenames:
                    found.append(dirpath)
            return sorted(found)

        # else treat as explicit list
        return [p for p in policy_folders if os.path.isdir(p)]

    def _make_policy_configs(self, policy_folders: List[str]) -> Dict[str, dict]:
        """
        Build policy_configs for ToolCallOrchestrator.
        Filters to policies compatible with ENV:
          - num_difficulties == 5
          - state_dim == 11
        """
        resolved = self._discover_policy_folders(policy_folders)
        print(f"\n[Policies] Scanning policy folders from: {policy_folders}")
        print(f"[Policies] Found {len(resolved)} candidate folders with config.json")

        compatible = []
        rejected = []

        for folder in resolved:
            cfg_path = os.path.join(folder, "config.json")
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
            except Exception as e:
                rejected.append((folder, f"config load failed: {e}"))
                continue

            sd = cfg.get("state_dim", None)
            nd = cfg.get("num_difficulties", None)

            if sd == 11 and nd == 5:
                compatible.append((folder, cfg))
            else:
                rejected.append((folder, f"incompatible: state_dim={sd}, num_difficulties={nd}"))

        print(f"[Policies] Compatible policies (state_dim=11,num_difficulties=5): {len(compatible)}")
        if compatible:
            for i, (folder, cfg) in enumerate(compatible[:20]):
                print(f"  - OK[{i}] {folder} | agent_type={cfg.get('agent_type')} objectives={cfg.get('objectives')} timestamp={cfg.get('timestamp')}")
            if len(compatible) > 20:
                print(f"  ... and {len(compatible) - 20} more")

        if rejected:
            print(f"[Policies] Rejected policies: {len(rejected)} (showing up to 10)")
            for folder, why in rejected[:10]:
                print(f"  - NO  {folder} | {why}")

        if not compatible:
            raise FileNotFoundError(
                "No compatible policies found under the provided folders.\n"
                "Need policies trained with state_dim=11 and num_difficulties=5."
            )

        # Build configs dict
        policy_configs: Dict[str, dict] = {}
        for i, (folder, cfg) in enumerate(compatible):
            key = f"policy_{i}"
            policy_configs[key] = {"folder_path": folder, "config": cfg}

        return policy_configs

    # ---------------------------
    # Orchestrator dynamic loader (no hard coding)
    # ---------------------------

    def _resolve_orchestrator_class(self, orchestrator_type: str):
        """
        Convention-based resolution:
          orchestrator.<type>_orchestrator module
          <Type>Orchestrator class (CamelCase)

        Examples:
          tool_call         -> orchestrator.tool_call_orchestrator.ToolCallOrchestrator
          context_based     -> orchestrator.context_based_orchestrator.ContextBasedOrchestrator
          reflection_based  -> orchestrator.reflection_based_orchestrator.ReflectionBasedOrchestrator
        """
        orch = (orchestrator_type or "tool_call").strip().lower()

        # typo guard
        if orch == "reflectin_based":
            orch = "reflection_based"

        allowed = {"tool_call", "context_based", "reflection_based"}
        if orch not in allowed:
            raise ValueError(f"Unknown orchestrator_type='{orchestrator_type}'. Allowed: {sorted(allowed)}")

        module_name = f"orchestrator.{orch}_orchestrator"
        class_name = "".join(part.capitalize() for part in orch.split("_")) + "Orchestrator"

        module = importlib.import_module(module_name)
        if not hasattr(module, class_name):
            raise ImportError(f"Module '{module_name}' does not define class '{class_name}'")

        return orch, module_name, class_name, getattr(module, class_name)

    def attach_orchestrator(
        self,
        policy_folders: List[str],
        model_name: str,
        objectives: List[str],
        verbose: bool = True,
        orchestrator_type: Optional[str] = None,  # optional override; defaults to settings.py
    ) -> None:
        # choose type from settings by default
        orch_type = orchestrator_type or SETTINGS_ORCHESTRATOR_TYPE

        llm = model_factory(model_name)
        llm_obj = llm
        llm_type = type(llm_obj).__name__
        llm_model = getattr(llm_obj, "model_name", None) or getattr(llm_obj, "model", None) or getattr(llm_obj, "name", None)
        print(f"[Orchestrator] LLM object type={llm_type}, model={llm_model}, model_name_arg={model_name}")

        # Resolve orchestrator class dynamically
        orch_key, module_name, class_name, OrchClass = self._resolve_orchestrator_class(orch_type)

        print("\n[Orchestrator] ===== Attaching orchestrator =====")
        print(f"[Orchestrator] settings.ORCHESTRATOR_TYPE={SETTINGS_ORCHESTRATOR_TYPE}")
        print(f"[Orchestrator] requested_orchestrator_type={orchestrator_type}")
        print(f"[Orchestrator] selected_orchestrator_type={orch_key}")
        print(f"[Orchestrator] resolved={module_name}.{class_name}")
        print(f"[Orchestrator] model_name={model_name}")
        print(f"[Orchestrator] objectives={objectives}")
        print("[Orchestrator] ----------------------------------")

        # Some orchestrators need policies (ToolCall), others may not.
        # We detect whether __init__ accepts policy_configs to avoid hard-coding.
        sig = inspect.signature(OrchClass.__init__)
        kwargs = dict(env=self.env, llm=llm, verbose=verbose, objectives=objectives)

        if "policy_configs" in sig.parameters:
            policy_configs = self._make_policy_configs(policy_folders)
            print(f"[Orchestrator] policy_folders={policy_folders}")
            print(f"[Orchestrator] num_policies={len(policy_configs)}")
            kwargs["policy_configs"] = policy_configs
        else:
            print("[Orchestrator] (This orchestrator does not use policy_configs.)")

        self.orchestrator = OrchClass(**kwargs)

        print("[Orchestrator] Attached successfully.\n")

    # ---------------------------
    # Learning loop API
    # ---------------------------

    def get_state_obs(self) -> np.ndarray:
        return self.env._get_obs()

    def select_action(self) -> Tuple[dict, dict, float]:
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not attached. Call attach_orchestrator() after pretest.")

        t0 = time.time()
        action_info, orch_info = self.orchestrator.select_action(self.get_state_obs())
        latency = time.time() - t0

        sel = orch_info.get("selected_strategy", None) if isinstance(orch_info, dict) else None
        print("\n[Step] Orchestrator decision")
        print(f"[Step] selected_strategy={sel} action_info={action_info} latency={latency:.3f}s")
        return action_info, orch_info, latency

    def get_questions_for_action(self, action: int) -> List[int]:
        qids = self.env._select_questions_by_action(int(action))

        # Print the batch with both UI level (1..6) and ENV level (1..5)
        print("[Step] Selected batch:")
        for qid in qids:
            ui_lvl = int(self.difficulties_display[qid].get("original_difficulty", 1))
            env_lvl = int(self.difficulties_env[qid].get("original_difficulty", 1))
            env_sd = float(self.difficulties_env[qid].get("scaled_difficulty", 0.2))
            dataset_id = self.questions[qid].get("id", qid)
            print(f"  - qid={qid} dataset_id={dataset_id} UI_level={ui_lvl} ENV_level={env_lvl} ENV_scaled={env_sd:.2f}")
        return qids

    def apply_learning_batch(self, action: int, qids: List[int], correctness: List[bool]) -> StepResult:
        if len(qids) != len(correctness):
            raise ValueError("apply_learning_batch: qids and correctness must have same length")

        mastery_before = float(self.env.mastery[0])

        # Log student answers being applied
        print("\n[Step] Applying student answers to env")
        print(f"[Step] action={int(action)} mastery_before={mastery_before:.3f}")
        for qid, correct in zip(qids, correctness):
            ui_lvl = int(self.difficulties_display[qid].get("original_difficulty", 1))
            env_lvl = int(self.difficulties_env[qid].get("original_difficulty", 1))
            print(f"  - qid={qid} UI_level={ui_lvl} ENV_level={env_lvl} correct={bool(correct)}")

        qinfo_list = []
        for qid, correct in zip(qids, correctness):
            qinfo = self.env.question_skills_difficulty_map[qid]
            skills = np.array([0], dtype=int)

            self.env.seen_materials.append((qid, qinfo["scaled_difficulty"], bool(correct)))

            qinfo_list.append({
                "question_id": qid,
                "skills": [self.topic],
                "original_difficulty": qinfo["original_difficulty"],   # env 1..5
                "scaled_difficulty": qinfo["scaled_difficulty"],       # env scaled
                "correct": bool(correct),
                "skills_tested": skills,
            })

            self.env._update_skill_difficulty_accuracy(
                qid, bool(correct), skills, qinfo["original_difficulty"]
            )

        rewards = self.env._calculate_batch_rewards(qinfo_list)
        self.env._update_mastery(qinfo_list)

        for qi in qinfo_list:
            self.env._update_failed_question(
                qi["question_id"],
                qi["skills_tested"],
                qi["correct"],
                qi["original_difficulty"],
                qi["scaled_difficulty"],
            )

        self.env._update_aptitude_cache()
        self.env._update_experience_cache()
        self.env._update_gap_cache()

        self.env.current_step += 1

        mastery_after = float(self.env.mastery[0])
        acc = float(sum(1 for c in correctness if c) / max(1, len(correctness)))

        print(f"[Step] mastery_after={mastery_after:.3f} acc={acc:.3f} rewards={rewards}\n")

        info = {
            "action": int(action),
            "selected_questions": list(map(int, qids)),
            "rolling_accuracy": acc,
            "mastery": {self.topic: round(mastery_after, 3)},
        }

        return StepResult(
            selected_qids=qids,
            rewards=rewards,
            mastery_before=mastery_before,
            mastery_after=mastery_after,
            rolling_accuracy=acc,
            info=info,
        )
