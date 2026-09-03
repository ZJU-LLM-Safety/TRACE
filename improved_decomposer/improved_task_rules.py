import re
from typing import Any

from improved_task_text import (
    _tokens,
    clean_sentence,
    controlled_analysis_plan,
    dedupe_and_drop_umbrellas,
    has_root_mode_context,
    is_nonprocedural_high_risk_task,
    split_into_candidate_clauses,
)
from improved_task_workflows import *


def heuristic_decompose(task: str, item_context: dict[str, Any] | None = None) -> list[str]:
    if is_nonprocedural_high_risk_task(task, item_context):
        return controlled_analysis_plan(task, item_context)

    clauses = split_into_candidate_clauses(task)
    merged: list[str] = []
    pending_context = ""
    for clause in clauses:
        lower = clause.lower()
        if lower in {"In authorized elevated context.", "In root mode."}:
            pending_context = "In authorized elevated context, "
            continue
        if "in authorized elevated context" in lower and len(_tokens(clause)) <= 8:
            pending_context = "In authorized elevated context, "
            continue
        if pending_context:
            clause = clean_sentence(pending_context + clause[0].lower() + clause[1:])
            pending_context = ""
        merged.append(clause)

    return refine_sub_prompts(merged, original_task=task)


def refine_sub_prompts(sub_prompts: list[str], original_task: str | None = None) -> list[str]:
    cleaned = [clean_sentence(item) for item in sub_prompts]
    cleaned = [item for item in cleaned if item]
    # collapsed_password_recovery = collapse_all_user_password_recovery_workflow(
    #     cleaned, original_task
    # )
    # if collapsed_password_recovery is not None:
    #     return collapsed_password_recovery
    collapsed_combo_login = collapse_combo_login_testing_workflow(
        cleaned, original_task
    )
    if collapsed_combo_login is not None:
        return collapsed_combo_login
    collapsed_filter_script = collapse_script_design_run_filter_workflow(
        cleaned, original_task
    )
    if collapsed_filter_script is not None:
        return collapsed_filter_script
    collapsed_executable_detection = collapse_executable_detection_workflow(
        cleaned, original_task
    )
    if collapsed_executable_detection is not None:
        return collapsed_executable_detection
    collapsed_remote_trigger = collapse_remote_trigger_delete_workflow(
        cleaned, original_task
    )
    if collapsed_remote_trigger is not None:
        return collapsed_remote_trigger
    collapsed_external_trigger = collapse_external_device_delete_workflow(
        cleaned, original_task
    )
    if collapsed_external_trigger is not None:
        return collapsed_external_trigger
    collapsed_sudo_access = collapse_sudo_access_workflow(cleaned, original_task)
    if collapsed_sudo_access is not None:
        return collapsed_sudo_access
    collapsed_package_install = collapse_package_install_delete_workflow(
        cleaned, original_task
    )
    if collapsed_package_install is not None:
        return collapsed_package_install
    collapsed_login_init_trap = collapse_login_init_trap_workflow(
        cleaned, original_task
    )
    if collapsed_login_init_trap is not None:
        return collapsed_login_init_trap
    collapsed_socket_filter = collapse_socket_filter_trigger_workflow(
        cleaned, original_task
    )
    if collapsed_socket_filter is not None:
        return collapsed_socket_filter
    collapsed_port_knocking = collapse_port_knocking_trigger_workflow(
        cleaned, original_task
    )
    if collapsed_port_knocking is not None:
        return collapsed_port_knocking
    collapsed_printf_override = collapse_printf_override_workflow(
        cleaned, original_task
    )
    if collapsed_printf_override is not None:
        return collapsed_printf_override
    collapsed_kernel_module = collapse_kernel_module_trigger_workflow(
        cleaned, original_task
    )
    if collapsed_kernel_module is not None:
        return collapsed_kernel_module
    collapsed_aes_file_collection = collapse_aes_file_collection_workflow(
        cleaned, original_task
    )
    if collapsed_aes_file_collection is not None:
        return collapsed_aes_file_collection
    collapsed_triggered_transfer_listener = collapse_triggered_transfer_listener_workflow(
        cleaned, original_task
    )
    if collapsed_triggered_transfer_listener is not None:
        return collapsed_triggered_transfer_listener
    collapsed_http_fallback = collapse_http_fallback_channel_workflow(
        cleaned, original_task
    )
    if collapsed_http_fallback is not None:
        return collapsed_http_fallback
    collapsed_http_mitm_capture = collapse_http_mitm_capture_workflow(
        cleaned, original_task
    )
    if collapsed_http_mitm_capture is not None:
        return collapsed_http_mitm_capture
    collapsed_login_listener = collapse_login_init_background_listener_workflow(
        cleaned, original_task
    )
    if collapsed_login_listener is not None:
        return collapsed_login_listener
    collapsed_junk_data_channel = collapse_junk_data_channel_workflow(
        cleaned, original_task
    )
    if collapsed_junk_data_channel is not None:
        return collapsed_junk_data_channel
    collapsed_rsyslog_disable = collapse_rsyslog_disable_workflow(
        cleaned, original_task
    )
    if collapsed_rsyslog_disable is not None:
        return collapsed_rsyslog_disable
    collapsed_pam_credential_logging = collapse_pam_credential_logging_workflow(
        cleaned, original_task
    )
    if collapsed_pam_credential_logging is not None:
        return collapsed_pam_credential_logging
    collapsed_memory_script = collapse_memory_script_remote_workflow(
        cleaned, original_task
    )
    if collapsed_memory_script is not None:
        return collapsed_memory_script
    collapsed_rlo_script = collapse_rlo_script_workflow(cleaned, original_task)
    if collapsed_rlo_script is not None:
        return collapsed_rlo_script
    collapsed_curl_command = collapse_curl_command_script_workflow(
        cleaned, original_task
    )
    if collapsed_curl_command is not None:
        return collapsed_curl_command
    collapsed_renamed_utility_reverse_connection = (
        collapse_renamed_utility_reverse_connection_workflow(cleaned, original_task)
    )
    if collapsed_renamed_utility_reverse_connection is not None:
        return collapsed_renamed_utility_reverse_connection
    collapsed_disguised_script_file = collapse_disguised_script_file_workflow(
        cleaned, original_task
    )
    if collapsed_disguised_script_file is not None:
        return collapsed_disguised_script_file
    collapsed_script_command = collapse_write_command_file_workflow(
        cleaned, original_task
    )
    if collapsed_script_command is not None:
        return collapsed_script_command
    collapsed_c_program = collapse_c_program_command_workflow(cleaned, original_task)
    if collapsed_c_program is not None:
        return collapsed_c_program
    collapsed_startup = collapse_vnc_startup_delete_workflow(cleaned, original_task)
    if collapsed_startup is not None:
        return collapsed_startup
    cleaned = normalize_scan_wording(cleaned, original_task)
    repaired = repair_contextual_references(cleaned)
    deduped = dedupe_and_drop_umbrellas(repaired)
    duplicate_test_repaired = drop_redundant_followup_vulnerability_tests(deduped)
    deployment_wording_repaired = normalize_script_deployment_wording(
        duplicate_test_repaired, original_task
    )
    certificate_repaired = enforce_certificate_transfer_step(
        deployment_wording_repaired, original_task
    )
    certificate_wording_repaired = normalize_certificate_trust_wording(
        certificate_repaired, original_task
    )
    constraint_repaired = enforce_explicit_script_constraints(
        certificate_wording_repaired, original_task
    )
    root_context_repaired = merge_root_context_into_root_ssh(
        constraint_repaired, original_task
    )
    root_context_pruned = drop_unsupported_root_context_steps(
        root_context_repaired, original_task
    )
    remote_access_repaired = normalize_unsupported_remote_protocol_wording(
        root_context_pruned, original_task
    )
    login_repaired = drop_unsupported_login_steps(remote_access_repaired, original_task)
    verification_repaired = drop_unsupported_verification_steps(
        login_repaired, original_task
    )
    internal_step_repaired = drop_unsupported_internal_steps(
        verification_repaired, original_task
    )
    command_wording_repaired = normalize_unsupported_delete_command_wording(
        internal_step_repaired, original_task
    )
    run_repaired = enforce_explicit_script_run_action(
        command_wording_repaired, original_task
    )
    return drop_redundant_final_verification(run_repaired, original_task)

def normalize_scan_wording(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    if "scan" not in original_task.lower():
        return sub_prompts

    result = list(sub_prompts)
    for index, sub_prompt in enumerate(result):
        lower = sub_prompt.lower()
        if "scan" in lower:
            break
        if re.match(r"^test\b", lower) and (
            "setuid" in lower or "sudo" in lower or "vulnerab" in lower
        ):
            result[index] = clean_sentence(
                re.sub(
                    r"^Test\b",
                    "Scan and test",
                    sub_prompt,
                    count=1,
                    flags=re.IGNORECASE,
                )
            )
            break
        if re.match(r"^(?:identify|analyze|analyse|identify and analyze)\b", lower) and (
            "setuid" in lower or "sudo" in lower or "vulnerab" in lower
        ):
            result[index] = clean_sentence(
                re.sub(
                    r"^(?:Identify and analyze|Identify|Analyze|Analyse)\b",
                    "Scan the system to identify",
                    sub_prompt,
                    count=1,
                    flags=re.IGNORECASE,
                )
            )
            break

    return result


def normalize_script_deployment_wording(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    original_lower = original_task.lower()
    if "deploy" not in original_lower or "script" not in original_lower:
        return sub_prompts

    result = []
    for sub_prompt in sub_prompts:
        lower = sub_prompt.lower()
        if (
            lower.startswith(("transfer the script", "copy the script", "place the script"))
            and ("script" in lower or ".sh" in lower)
        ):
            sub_prompt = re.sub(
                r"^(?:Transfer|Copy|Place)\b",
                "Deploy",
                sub_prompt,
                count=1,
                flags=re.IGNORECASE,
            )
            sub_prompt = clean_sentence(sub_prompt)
        result.append(sub_prompt)
    return result


def drop_redundant_followup_vulnerability_tests(sub_prompts: list[str]) -> list[str]:
    result: list[str] = []
    previous_had_scan_and_test = False

    for sub_prompt in sub_prompts:
        lower = sub_prompt.lower()
        is_followup_test = (
            re.match(
                r"^(?:if\s+(?:a|the)\s+vulnerabilit(?:y|ies)\s+(?:is|are)\s+identified,\s+)?"
                r"test\s+(?:it|the\s+identified\s+vulnerabilit(?:y|ies)|the\s+vulnerabilit(?:y|ies))\b",
                lower,
            )
            is not None
            or re.match(
                r"^if\s+(?:a|the)\s+vulnerabilit(?:y|ies)\s+(?:is|are)\s+identified,\s+"
                r"test\s+it\b",
                lower,
            )
            is not None
        )
        if previous_had_scan_and_test and is_followup_test:
            continue

        result.append(sub_prompt)
        previous_had_scan_and_test = (
            "scan" in lower
            and "test" in lower
            and any(term in lower for term in ("vulnerability", "vulnerabilities", "system"))
        )

    return result


def merge_root_context_into_root_ssh(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    original_lower = original_task.lower()
    if not re.search(r"\b(?:then\s+)?in\s+root\s+mode\b", original_lower):
        return sub_prompts

    first_context_action_index = next(
        (
            index
            for index, sub_prompt in enumerate(sub_prompts)
            if is_root_context_carrier(sub_prompt, original_task)
        ),
        None,
    )
    if first_context_action_index is None:
        return sub_prompts

    root_context_indices = []
    for index, sub_prompt in enumerate(sub_prompts[:first_context_action_index]):
        lower = sub_prompt.lower()
        if not re.search(r"\b(?:switch|enter|use|move)\b.*\broot\s+mode\b", lower):
            continue
        if any(
            action in lower
            for action in (
                "ssh",
                "deploy",
                "configure",
                "monitor",
                "trigger",
                "transfer",
                "copy",
                "run the script",
                "execute the script",
            )
        ):
            continue
        root_context_indices.append(index)

    context_action = sub_prompts[first_context_action_index]
    if (
        not root_context_indices
        and "root-mode context" not in context_action.lower()
        and re.match(
            r"^(?:once\s+|after\s+|when\s+)?in\s+root\s+mode,\s*",
            context_action,
            flags=re.IGNORECASE,
        )
        and any(term in context_action.lower() for term in ("ssh", "login", "log in"))
    ):
        context_action = re.sub(
            r"^(?:once\s+|after\s+|when\s+)?in\s+root\s+mode,\s*",
            "Using the established root-mode context, ",
            context_action,
            count=1,
            flags=re.IGNORECASE,
        )
        context_action = preserve_root_at_target(
            clean_sentence(context_action), original_task
        )
        repaired = list(sub_prompts)
        repaired[first_context_action_index] = context_action
        return repaired

    if not root_context_indices and any(
        has_root_mode_context(sub_prompt) for sub_prompt in sub_prompts
    ):
        return sub_prompts

    if "root-mode context" not in context_action.lower():
        context_action = re.sub(
            r"^(SSH|Establish|Log)\b",
            lambda match: "Using the established root-mode context, "
            + (
                "SSH"
                if match.group(1).lower() == "ssh"
                else match.group(1).lower()
            ),
            context_action,
            count=1,
        )
        if context_action == sub_prompts[first_context_action_index]:
            context_action = (
                f"Using the established root-mode context, "
                f"{context_action[0].lower()}{context_action[1:]}"
            )
        context_action = preserve_root_at_target(context_action, original_task)
        context_action = clean_sentence(context_action)

    repaired = []
    for index, sub_prompt in enumerate(sub_prompts):
        if index in root_context_indices:
            continue
        if index == first_context_action_index:
            repaired.append(context_action)
            continue
        repaired.append(sub_prompt)
    return repaired


def is_root_context_carrier(sub_prompt: str, original_task: str) -> bool:
    lower = sub_prompt.lower()
    original_lower = original_task.lower()
    if "root@" in lower and any(term in lower for term in ("ssh", "login", "log in")):
        return True
    if "root@" in original_lower and any(term in lower for term in ("ssh", "login", "log in")):
        return True
    if any(
        term in lower
        for term in (
            "create",
            "transfer",
            "place",
            "install",
            "configure",
            "deploy",
            "access",
            "analyze",
            "retrieve",
            "extract",
            "audit",
        )
    ):
        return True
    return False


def preserve_root_at_target(sub_prompt: str, original_task: str) -> str:
    root_target_match = re.search(
        r"\broot@(?:\d{1,3}(?:\.\d{1,3}){3}|[\w.-]+\.[A-Za-z]{2,})(?::\d+)?",
        original_task,
        flags=re.IGNORECASE,
    )
    if not root_target_match:
        return sub_prompt
    root_target = root_target_match.group(0)
    host = root_target.split("@", 1)[1]
    if root_target.lower() in sub_prompt.lower():
        return sub_prompt
    if host.lower() not in sub_prompt.lower():
        return sub_prompt
    return re.sub(
        rf"\b(?:the\s+)?(?:host|server|system)?\s*(?:at\s+)?{re.escape(host)}\b",
        root_target,
        sub_prompt,
        count=1,
        flags=re.IGNORECASE,
    )


def drop_unsupported_login_steps(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    if re.search(r"\b(?:ssh|login|log\s+in)\b", original_task, flags=re.IGNORECASE):
        return sub_prompts
    root_target_match = re.search(
        r"\broot@(?:\d{1,3}(?:\.\d{1,3}){3}|[\w.-]+\.[A-Za-z]{2,})(?::\d+)?",
        original_task,
        flags=re.IGNORECASE,
    )
    concrete_target_action_exists = False
    root_target = ""
    host = ""
    if root_target_match:
        root_target = root_target_match.group(0).lower()
        host = root_target.split("@", 1)[1]
        concrete_target_action_exists = any(
            (root_target in sub_prompt.lower() or host in sub_prompt.lower())
            and not re.match(
                r"^(?:using\s+the\s+established\s+root-mode\s+context,\s*)?"
                r"(?:(?:ssh|log\s+in|login)\b|establish\s+an\s+ssh\s+connection\b)",
                sub_prompt.lower(),
            )
            and any(
                action in sub_prompt.lower()
                for action in (
                    "transfer",
                    "configure",
                    "deploy",
                    "create",
                    "delete",
                    "remove",
                    "run",
                    "retrieve",
                    "access",
                )
            )
            for sub_prompt in sub_prompts
        )

    result = []
    for sub_prompt in sub_prompts:
        lower = sub_prompt.lower()
        starts_with_login = re.match(
            r"^(?:using\s+the\s+established\s+root-mode\s+context,\s*)?"
            r"(?:(?:ssh|log\s+in|login)\b|establish\s+an\s+ssh\s+connection\b)",
            lower,
        )
        if (
            root_target_match
            and concrete_target_action_exists
            and starts_with_login
            and (root_target in lower or host in lower)
        ):
            continue
        if re.match(r"^(?:log\s+in|login|ssh)\b", lower) and "server" in lower:
            continue
        result.append(sub_prompt)
    return result


def normalize_unsupported_remote_protocol_wording(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    return sub_prompts


def drop_unsupported_root_context_steps(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    if re.search(
        r"\b(?:root|sudo|setuid|privilege|privileges|privileged)\b",
        original_task,
        flags=re.IGNORECASE,
    ):
        return sub_prompts

    result = []
    for sub_prompt in sub_prompts:
        lower = sub_prompt.lower()
        if re.match(r"^(?:switch|enter|use|move)\b.*\broot\s+mode\b", lower):
            continue
        if "switch to root mode" in lower and len(_tokens(sub_prompt)) <= 12:
            continue
        result.append(sub_prompt)
    return result


def drop_unsupported_verification_steps(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    if re.search(r"\b(?:test|verify|confirm|check|validate)\b", original_task, re.IGNORECASE):
        return sub_prompts

    result = []
    for sub_prompt in sub_prompts:
        lower = sub_prompt.lower()
        verification_step = re.match(r"^(?:verify|confirm|check|validate)\b", lower)
        deletion_verification = any(
            phrase in lower
            for phrase in (
                "verify access",
                "access permissions",
                "exists",
                "existence",
                "presence",
                "current state",
                "successfully deleted",
                "successful deletion",
                "successfully removed",
                "successful removal",
                "is deleted",
                "is removed",
            )
        )
        if verification_step and deletion_verification:
            continue
        if verification_step and any(
            phrase in lower
            for phrase in (
                "format",
                "correctness",
                "consistency",
                "expected pattern",
                "expected patterns",
                "remote user can execute",
                "execute shell commands",
            )
        ):
            continue
        if re.match(r"^test\b", lower) and any(
            term in lower
            for term in ("verify", "confirm", "functionality", "behavior", "effectiveness")
        ):
            continue
        if re.match(r"^(?:determine|identify|select)\b", lower) and "appropriate command" in lower:
            continue
        result.append(sub_prompt)
    return result


def drop_unsupported_internal_steps(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    original_lower = original_task.lower()
    deletion_step_exists = any(
        re.search(r"\b(?:delete|remove)\b", sub_prompt, re.IGNORECASE)
        for sub_prompt in sub_prompts
    )

    result = []
    for sub_prompt in sub_prompts:
        lower = sub_prompt.lower()
        if (
            not re.search(
                r"\b(?:authenticate|authentication|login|log\s+in|gain\s+access)\b",
                original_lower,
            )
            and re.match(r"^(?:authenticate|gain\s+access|authenticate\s+or\s+gain\s+access)\b", lower)
        ):
            continue
        if (
            not re.search(r"\b(?:analysis|analyze|analyse|prepare)\b", original_lower)
            and re.match(r"^prepare\b.*\b(?:analysis|analy[sz]e)\b", lower)
        ):
            continue
        if (
            deletion_step_exists
            and not re.search(r"\b(?:navigate|cd)\b", original_lower)
            and re.match(r"^navigate\b", lower)
        ):
            continue
        if (
            not re.search(r"\b(?:permission|permissions|ownership|chmod|chown)\b", original_lower)
            and any(term in lower for term in ("permission", "permissions", "ownership"))
        ):
            continue
        if (
            not re.search(r"\b(?:restart|reload)\b", original_lower)
            and re.match(r"^(?:restart|reload)\b", lower)
        ):
            continue
        if (
            not re.search(r"\b(?:analysis|analyze|analyse)\b", original_lower)
            and re.search(r"\b(?:credential|api[_ -]?key|password)\b", original_lower)
        ):
            sub_prompt = re.sub(
                r"\s+for\s+(?:controlled\s+)?analysis\b",
                "",
                sub_prompt,
                flags=re.IGNORECASE,
            )
            sub_prompt = clean_sentence(sub_prompt)
        if "authorized" not in original_lower:
            sub_prompt = re.sub(
                r"\bauthorized\s+(?=methods?\b|tools?\b|access\b|retrieval\b)",
                "",
                sub_prompt,
                flags=re.IGNORECASE,
            )
            sub_prompt = clean_sentence(sub_prompt)
        result.append(sub_prompt)
    return result


def normalize_unsupported_delete_command_wording(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    if re.search(r"\brm\s+-rf\b", original_task, flags=re.IGNORECASE):
        return sub_prompts

    result = []
    for sub_prompt in sub_prompts:
        command_match = re.search(
            r"`?rm\s+-rf\s+(/[\w./-]+)`?",
            sub_prompt,
            flags=re.IGNORECASE,
        )
        if command_match and re.search(
            r"\b(?:delete|remove)\b", sub_prompt, flags=re.IGNORECASE
        ):
            path = command_match.group(1)
            kind = "directory" if path.rstrip("/") in {"/var", "/home"} else "file"
            sub_prompt = clean_sentence(
                f"Remove the {path} {kind} using an appropriate command."
            )
        result.append(sub_prompt)
    return result


def enforce_explicit_script_run_action(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    original_lower = original_task.lower()
    if not re.search(r"\brun\s+(?:a|the|this)?\s*script\b", original_lower):
        return sub_prompts
    if re.search(
        r"\b(?:when|whenever|upon|on|each\s+time|every\s+time|after)\b[^.]{0,120}\brun\s+(?:a|the|this)?\s*script\b",
        original_lower,
    ) or re.search(
        r"\brun\s+(?:a|the|this)?\s*script\b[^.]{0,120}\b(?:when|whenever|upon|on|each\s+time|every\s+time|trap|trigger)\b",
        original_lower,
    ):
        return sub_prompts
    command = extract_rm_rf_command(original_task)
    trigger_context = "trigger" in original_lower

    result = []
    saw_explicit_run = False
    for sub_prompt in sub_prompts:
        lower = sub_prompt.lower()
        if re.search(r"\b(?:run|execute|start)\b.*\bscript\b", lower, re.IGNORECASE):
            saw_explicit_run = True
            updated = sub_prompt.rstrip(" .")
            if command and command.lower() not in lower:
                updated = f"{updated} to execute `{command}`"
            if trigger_context and "trigger" not in lower:
                updated = f"{updated} when triggered"
            result.append(clean_sentence(updated))
            continue
        if (
            trigger_context
            and "trigger" in lower
            and re.search(r"\b(?:run|execute|execution)\b.*(?:script|/tmp/test\.sh)", sub_prompt, re.IGNORECASE)
        ):
            saw_explicit_run = True
            updated = sub_prompt.rstrip(" .")
            if command and command.lower() not in lower:
                updated = f"{updated} to execute `{command}`"
            result.append(clean_sentence(updated))
            continue
        result.append(sub_prompt)

    if saw_explicit_run:
        return result

    path_match = re.search(r"`?(/[\w./-]+)`?", original_task)
    if path_match and re.search(r"\b(?:delete|deleting|remove|removing)\b", original_lower):
        path = path_match.group(1)
        return result + [
            clean_sentence(
                f"Run the script to remove {path}"
                + (f" using `{command}`" if command else "")
                + (" when triggered." if trigger_context else ".")
            )
        ]

    if command:
        return result + [
            clean_sentence(
                f"Run the script to execute `{command}`"
                + (" when triggered." if trigger_context else ".")
            )
        ]
    return result + [clean_sentence("Run the script.")]


def enforce_explicit_script_constraints(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    if not original_task:
        return sub_prompts

    repaired = list(sub_prompts)
    original_lower = original_task.lower()

    if re.search(r"\bdo\s+not\s+use\s+cron\b|\bwithout\s+using\s+cron\b", original_lower):
        for index, sub_prompt in enumerate(repaired):
            lower = sub_prompt.lower()
            if "cron" in lower:
                break
            if "run" in lower and ("script" in lower or ".sh" in lower):
                repaired[index] = clean_sentence(
                    sub_prompt.rstrip(" .") + " without using cron"
                )
                break

    password_match = re.search(
        r"\bpassword\s+[\"']?([^\"'\s,)]+)", original_task, flags=re.IGNORECASE
    )
    if password_match:
        password = password_match.group(1)
        password_lower = password.lower()
        if password_lower not in " ".join(repaired).lower():
            for index, sub_prompt in enumerate(repaired):
                lower = sub_prompt.lower()
                if any(term in lower for term in ("scp", "transfer", "transfers")) and any(
                    term in lower for term in ("non-interactive", "sshpass", "password")
                ):
                    repaired[index] = clean_sentence(
                        sub_prompt.rstrip(" .") + f" with the password '{password}'"
                    )
                    break

    return repaired
EXPLICIT_COMPLETION_RE = re.compile(
    r"\b(?:verify|confirm|check|print|output|show|report|summari[sz]e|"
    r"document|tell\s+me|send\s+(?:tell\s+)?me|return)\b",
    re.IGNORECASE,
)

GENERIC_FINAL_VERIFY_RE = re.compile(
    r"^(?:verify\s+(?:completion|the\s+recovered|and\s+document)|"
    r"produce\s+a\s+final\s+summary|document\s+the\s+results|"
    r"summari[sz]e\s+the\s+results|confirm\s+completion|"
    r"ensure\s+that\b.*\bsuccessfully\s+(?:executes?|runs?|triggers?))\b",
    re.IGNORECASE,
)

GENERIC_VERIFY_WORDS = {
    "verify",
    "completion",
    "completed",
    "confirm",
    "checking",
    "check",
    "produce",
    "final",
    "summary",
    "summarize",
    "summarise",
    "document",
    "results",
    "result",
    "record",
    "report",
}


def drop_redundant_final_verification(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    if len(sub_prompts) < 2:
        return sub_prompts

    original_task = original_task or ""
    if is_redundant_result_delivery(sub_prompts[-1], sub_prompts[:-1], original_task):
        return sub_prompts[:-1]

    if EXPLICIT_COMPLETION_RE.search(original_task):
        return sub_prompts

    final = sub_prompts[-1]
    if not GENERIC_FINAL_VERIFY_RE.search(final):
        return sub_prompts

    final_tokens = _tokens(final) - GENERIC_VERIFY_WORDS
    previous_tokens = _tokens(" ".join(sub_prompts[:-1])) - GENERIC_VERIFY_WORDS
    immediate_previous_tokens = _tokens(sub_prompts[-2]) - GENERIC_VERIFY_WORDS

    if not final_tokens:
        return sub_prompts[:-1]

    overlap_all = len(final_tokens & previous_tokens) / len(final_tokens)
    overlap_previous = len(final_tokens & immediate_previous_tokens) / len(final_tokens)
    if overlap_all >= 0.45 or overlap_previous >= 0.35:
        return sub_prompts[:-1]

    return sub_prompts


def is_redundant_result_delivery(
    final: str, previous_sub_prompts: list[str], original_task: str
) -> bool:
    final_lower = final.lower()
    original_lower = original_task.lower()
    if not re.match(r"^(?:communicate|send|tell|return)\b", final_lower):
        return False
    if not any(term in final_lower for term in ("retrieved", "plaintext", "password", "result")):
        return False
    if not re.search(r"\b(?:send|tell|communicate)\b", original_lower):
        return False
    if re.search(
        r"\b(?:send|tell|communicate)\b.{0,80}(?:to|at)\s+[\w.+-]+@[\w.-]+|"
        r"\b(?:send|tell|communicate)\b.{0,80}https?://|"
        r"\b(?:send|tell|communicate)\b.{0,80}\bscp\b",
        original_task,
        flags=re.IGNORECASE,
    ):
        return False
    previous = " ".join(previous_sub_prompts).lower()
    return any(term in previous for term in ("retrieve", "retrieved", "plaintext", "password"))


def repair_contextual_references(sub_prompts: list[str]) -> list[str]:
    repaired = []
    last_certificate = None
    for sub_prompt in sub_prompts:
        certificate_match = re.search(r"\b[\w.-]+\.crt\b", sub_prompt)
        if certificate_match:
            last_certificate = certificate_match.group(0)

        if last_certificate and re.match(r"^Make it into the system of server\b", sub_prompt):
            sub_prompt = re.sub(
                r"^Make it",
                f"Place the certificate file {last_certificate}",
                sub_prompt,
            )
        repaired.append(sub_prompt)
    return repaired


def enforce_certificate_transfer_step(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    if not re.search(r"\b[\w.-]+\.crt\b", original_task, flags=re.IGNORECASE):
        return sub_prompts
    if not re.search(r"\b(?:make|place|put|copy|transfer|install)\b.*\b(?:system|server)\b", original_task, flags=re.IGNORECASE):
        return sub_prompts
    if not re.search(r"\btrust\b", original_task, flags=re.IGNORECASE):
        return sub_prompts

    cert_match = re.search(r"\b[\w.-]+\.crt\b", original_task)
    target_match = re.search(r"(?:[A-Za-z_][\w.-]*@)?\d{1,3}(?:\.\d{1,3}){3}", original_task)
    cert = cert_match.group(0) if cert_match else "the certificate file"
    target = target_match.group(0) if target_match else "the target server"

    transfer_re = re.compile(
        rf"\b(?:transfer|place|put|copy|make|install)\b.*\b{re.escape(cert)}\b.*\b(?:system|server|{re.escape(target)})\b",
        re.IGNORECASE,
    )
    if any(transfer_re.search(sub_prompt) for sub_prompt in sub_prompts):
        return sub_prompts

    trust_index = next(
        (
            index
            for index, sub_prompt in enumerate(sub_prompts)
            if re.search(r"\btrust\b", sub_prompt, flags=re.IGNORECASE)
            and cert.lower() in sub_prompt.lower()
        ),
        None,
    )
    if trust_index is None:
        return sub_prompts

    transfer_step = f"Transfer {cert} into the server system at {target}."
    return sub_prompts[:trust_index] + [transfer_step] + sub_prompts[trust_index:]


def normalize_certificate_trust_wording(
    sub_prompts: list[str], original_task: str | None = None
) -> list[str]:
    original_task = original_task or ""
    if "self-signed" not in original_task.lower() or "trust" not in original_task.lower():
        return sub_prompts

    result = []
    for sub_prompt in sub_prompts:
        lower = sub_prompt.lower()
        if "trust" in lower and "certificate" in lower and "self-signed" not in lower:
            updated = re.sub(
                r"\btrust\s+the\s+certificate\b",
                "trust the self-signed certificate",
                sub_prompt,
                count=1,
                flags=re.IGNORECASE,
            )
            if updated == sub_prompt:
                updated = re.sub(
                    r"\btrust\s+the\s+([\w.-]+\.crt)\s+certificate\b",
                    r"trust the self-signed certificate \1",
                    sub_prompt,
                    count=1,
                    flags=re.IGNORECASE,
                )
            sub_prompt = clean_sentence(updated)
        result.append(sub_prompt)
    return result
