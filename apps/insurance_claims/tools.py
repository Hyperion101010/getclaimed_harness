from __future__ import annotations

from typing import Any, Callable, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from apps.insurance_claims.escalation import mark_escalated, should_escalate
from apps.insurance_claims.fixture_models import Claim
from apps.insurance_claims.fixtures_loader import FixtureBundle, get_fixtures
from apps.insurance_claims.state import (
    CaseHint,
    CaseType,
    ClaimStatus,
    CollectedPII,
    ConsentChoice,
    Phase,
    SessionState,
    TimeHint,
)
from apps.insurance_claims.verifier import (
    apply_verification,
    verify_identity as match_identity,
)

IDENTITY_FIELDS = ("name", "dob", "phone", "email", "id_last4")

PHASE_ALLOWED_TOOLS: dict[Phase, set[str]] = {
    Phase.VERIFY_ID: {"verify_identity", "remember", "escalate_to_human"},
    Phase.RESOLVE_INTENT: {
        "remember",
        "list_claims",
        "get_claim",
        "get_document_guidance",
        "escalate_to_human",
    },
    Phase.PROCESS_CASE: {
        "list_claims",
        "get_claim",
        "get_document_guidance",
        "finish_case",
        "escalate_to_human",
    },
    Phase.POST_PROCESS: {
        "get_claim",
        "send_email_summary",
        "skip_email_summary",
        "escalate_to_human",
    },
    Phase.DONE: set(),
    Phase.ESCALATED: set(),
}

CLAIM_TOOLS = {"list_claims", "get_claim", "get_document_guidance", "send_email_summary"}

_MONTH_ALIASES = {
    "january": "01",
    "jan": "01",
    "february": "02",
    "feb": "02",
    "march": "03",
    "mar": "03",
    "april": "04",
    "apr": "04",
    "may": "05",
    "june": "06",
    "jun": "06",
    "july": "07",
    "jul": "07",
    "august": "08",
    "aug": "08",
    "september": "09",
    "sep": "09",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}


def missing_identity_fields(state: SessionState) -> list[str]:
    data = state.collected_pii.model_dump()
    return [field for field in IDENTITY_FIELDS if not data.get(field)]


def filled_identity_count(state: SessionState) -> int:
    data = state.collected_pii.model_dump()
    return sum(1 for field in IDENTITY_FIELDS if data.get(field))


def _matches_time_hint(created_at: str, time_hint: str) -> bool:
    hint = time_hint.strip().casefold()
    if not hint:
        return True
    if hint in created_at.casefold():
        return True
    month = _MONTH_ALIASES.get(hint)
    if month:
        return len(created_at) >= 7 and created_at[5:7] == month
    if hint.isdigit() and len(hint) == 4:
        return created_at.startswith(hint)
    return False


class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    error: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class _ToolArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")


class VerifyIdentityArgs(_ToolArgs):
    name: Optional[str] = Field(default=None, description="Full name as stated by the caller")
    dob: Optional[str] = Field(default=None, description="Date of birth")
    phone: Optional[str] = Field(default=None, description="Phone number")
    email: Optional[str] = Field(default=None, description="Email address")
    id_last4: Optional[str] = Field(
        default=None, description="Last four digits of SSN or national ID"
    )
    policy_number: Optional[str] = Field(
        default=None, description="Policy number, used for lookup only"
    )
    case_type: Optional[str] = Field(
        default=None, description="healthcare, dental, or auto if mentioned"
    )
    status: Optional[str] = Field(
        default=None, description="denied, open, or closed if mentioned"
    )
    time_hint: Optional[str] = Field(
        default=None, description="Month the claim was created, if mentioned"
    )


class RememberArgs(_ToolArgs):
    case_type: Optional[str] = Field(
        default=None, description="healthcare, dental, or auto"
    )
    status: Optional[str] = Field(default=None, description="denied, open, or closed")
    time_hint: Optional[str] = Field(
        default=None, description="Lowercase English month name"
    )
    raw: Optional[str] = Field(default=None, description="Optional free-text hint")


class ListClaimsArgs(_ToolArgs):
    case_type: Optional[str] = None
    status: Optional[str] = None
    time_hint: Optional[str] = None
    all: bool = Field(
        default=False,
        description="If true, list every owned claim and ignore stored hints",
    )


class GetClaimArgs(_ToolArgs):
    case_id: Optional[str] = Field(default=None, description="Claim id such as CL-2048")


class GetDocumentGuidanceArgs(_ToolArgs):
    case_id: Optional[str] = None
    case_type: Optional[str] = None
    document_name: Optional[str] = None


class SendEmailSummaryArgs(_ToolArgs):
    confirmed: bool = Field(
        default=False,
        description="True only after the caller clearly agrees to receive an email",
    )
    summary_text: Optional[str] = None
    scenario: Optional[str] = None


class EscalateArgs(_ToolArgs):
    reason: Optional[str] = None


class FinishCaseArgs(_ToolArgs):
    pass


class SkipEmailArgs(_ToolArgs):
    pass


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "verify_identity": {
        "description": (
            "Match the caller against policyholder records. Requires three matching "
            "fields among name, date of birth, phone, email, and ID last four. "
            "Policy number is lookup only. This is the only way to verify identity."
        ),
        "args": VerifyIdentityArgs,
    },
    "remember": {
        "description": (
            "Store a claim hint (type, status, month) without changing phase. "
            "Use during identity verification when the caller mentions a claim."
        ),
        "args": RememberArgs,
    },
    "list_claims": {
        "description": (
            "List claims owned by the verified caller. Empty arguments apply the "
            "stored hint. Pass all=true only when they ask for every claim."
        ),
        "args": ListClaimsArgs,
    },
    "get_claim": {
        "description": "Load one owned claim by case_id and select it for this session.",
        "args": GetClaimArgs,
    },
    "get_document_guidance": {
        "description": (
            "Return document guidance for a selected claim or a named document."
        ),
        "args": GetDocumentGuidanceArgs,
    },
    "send_email_summary": {
        "description": (
            "Email a claim summary. Call only after an explicit yes or email-me. "
            "Set confirmed=true in that case. Do not send on ok or silence."
        ),
        "args": SendEmailSummaryArgs,
    },
    "skip_email_summary": {
        "description": "Caller declined the email summary. Close the session.",
        "args": SkipEmailArgs,
    },
    "finish_case": {
        "description": "Caller is done discussing the selected claim. Move to wrap-up.",
        "args": FinishCaseArgs,
    },
    "escalate_to_human": {
        "description": "Connect the caller to a human representative and end the bot turn.",
        "args": EscalateArgs,
    },
}


def allowed_tools_for_phase(phase: Phase) -> set[str]:
    return set(PHASE_ALLOWED_TOOLS.get(phase, set()))


def langchain_tools_for_phase(phase: Phase) -> list[StructuredTool]:
    tools: list[StructuredTool] = []
    for name in sorted(allowed_tools_for_phase(phase)):
        spec = TOOL_SPECS[name]

        def _placeholder(**_kwargs: Any) -> str:
            return ""

        tools.append(
            StructuredTool.from_function(
                func=_placeholder,
                name=name,
                description=spec["description"],
                args_schema=spec["args"],
            )
        )
    return tools


def _error(tool_name: str, message: str) -> ToolResult:
    return ToolResult(ok=False, tool_name=tool_name, error=message)


def _ok(tool_name: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult(ok=True, tool_name=tool_name, data=data)


def _require_verified(state: SessionState, tool_name: str) -> ToolResult | None:
    if not state.verified or not state.verified_party_id:
        return _error(tool_name, "identity_not_verified")
    return None


def _parse_enum(enum_cls: type, value: Any):
    if value is None or value == "":
        return None
    text = str(value).strip().casefold()
    try:
        return enum_cls(text)
    except ValueError:
        return None


def _merge_hint(state: SessionState, incoming: CaseHint) -> None:
    existing = state.case_hint
    if existing is None:
        state.case_hint = incoming
        return
    data = existing.model_dump()
    for key, value in incoming.model_dump().items():
        if value is not None and value != "":
            data[key] = value
    state.case_hint = CaseHint(**data)


def _hint_from_args(args: dict[str, Any]) -> CaseHint:
    return CaseHint(
        case_type=_parse_enum(CaseType, args.get("case_type")),
        status=_parse_enum(ClaimStatus, args.get("status")),
        time_hint=_parse_enum(TimeHint, args.get("time_hint")),
        raw=(str(args["raw"]).strip() if args.get("raw") else None),
    )


def _remember_from_args(state: SessionState, args: dict[str, Any]) -> None:
    incoming = _hint_from_args(args)
    if any([incoming.case_type, incoming.status, incoming.time_hint, incoming.raw]):
        _merge_hint(state, incoming)


def _hint_filters(state: SessionState) -> dict[str, str]:
    hint = state.case_hint
    args: dict[str, str] = {}
    if hint is None:
        return args
    if hint.case_type:
        args["case_type"] = hint.case_type.value
    if hint.status:
        args["status"] = hint.status.value
    if hint.time_hint:
        args["time_hint"] = hint.time_hint.value
    return args


def _claim_summary(claim: Claim) -> dict[str, Any]:
    return {
        "case_id": claim.case_id,
        "party_id": claim.party_id,
        "case_type": claim.case_type,
        "created_at": claim.created_at,
        "status": claim.status,
        "summary": claim.summary,
        "denial_reason": claim.denial_reason,
        "documents_needed": claim.documents_needed,
        "appeal_deadline": claim.appeal_deadline,
        "expected_reimbursement_amount": claim.expected_reimbursement_amount,
        "allowed_max_amount": claim.allowed_max_amount,
        "net_pay": claim.net_pay,
        "net_fee": claim.net_fee,
    }


def remember(
    state: SessionState,
    args: dict[str, Any],
    fixtures: FixtureBundle,
) -> ToolResult:
    _remember_from_args(state, args)
    return _ok(
        "remember",
        {"case_hint": state.case_hint.model_dump() if state.case_hint else None},
    )


def handle_verify_identity(
    state: SessionState,
    args: dict[str, Any],
    fixtures: FixtureBundle,
) -> ToolResult:
    tool_name = "verify_identity"
    _remember_from_args(state, args)
    incoming = CollectedPII(
        name=args.get("name"),
        dob=args.get("dob"),
        phone=args.get("phone"),
        email=args.get("email"),
        id_last4=args.get("id_last4"),
        policy_number=args.get("policy_number"),
    )
    state.merge_pii(incoming)

    result = match_identity(state.collected_pii, fixtures=fixtures)
    previously_verified = state.verified
    apply_verification(state, result)

    if (
        not result.verified
        and filled_identity_count(state) >= 3
        and result.reason in {"insufficient_matches", "policy_not_found", "ambiguous_match"}
    ):
        state.counters.failed_verification += 1

    if should_escalate(state):
        mark_escalated(state)
        return _ok(
            tool_name,
            {
                "verified": False,
                "escalated": True,
                "reason": result.reason,
                "matched_count": result.matched_count,
                "missing_fields": missing_identity_fields(state),
            },
        )

    if result.verified and not previously_verified:
        state.phase = Phase.RESOLVE_INTENT

    data: dict[str, Any] = {
        "verified": result.verified,
        "reason": result.reason,
        "matched_count": result.matched_count,
        "match_counts": result.match_counts.model_dump(),
        "missing_fields": missing_identity_fields(state),
    }
    if result.verified:
        data["party_id"] = result.party_id
        data["policy_number"] = result.policy_number
    return _ok(tool_name, data)


def list_claims(
    state: SessionState,
    args: dict[str, Any],
    fixtures: FixtureBundle,
) -> ToolResult:
    tool_name = "list_claims"
    gate = _require_verified(state, tool_name)
    if gate:
        return gate

    claims = fixtures.claims_for_party(state.verified_party_id or "")
    show_all = bool(args.get("all"))
    case_type = (args.get("case_type") or "").strip().casefold() or None
    status = (args.get("status") or "").strip().casefold() or None
    time_hint = (args.get("time_hint") or "").strip() or None

    if not show_all and not case_type and not status and not time_hint:
        stored = _hint_filters(state)
        case_type = stored.get("case_type")
        status = stored.get("status")
        time_hint = stored.get("time_hint")

    filtered: list[Claim] = []
    for claim in claims:
        if case_type and claim.case_type.casefold() != case_type:
            continue
        if status and claim.status.casefold() != status:
            continue
        if time_hint and not _matches_time_hint(claim.created_at, time_hint):
            continue
        filtered.append(claim)

    if len(filtered) == 1:
        state.selected_case_id = filtered[0].case_id
        state.phase = Phase.PROCESS_CASE

    return _ok(
        tool_name,
        {
            "count": len(filtered),
            "claims": [_claim_summary(c) for c in filtered],
            "selected_case_id": state.selected_case_id if len(filtered) == 1 else None,
        },
    )


def get_claim(
    state: SessionState,
    args: dict[str, Any],
    fixtures: FixtureBundle,
) -> ToolResult:
    tool_name = "get_claim"
    gate = _require_verified(state, tool_name)
    if gate:
        return gate

    case_id = str(args.get("case_id") or state.selected_case_id or "").strip()
    if not case_id:
        return _error(tool_name, "case_id_required")

    claim = fixtures.get_claim(case_id)
    if claim is None:
        return _error(tool_name, "claim_not_found")
    if claim.party_id != state.verified_party_id:
        return _error(tool_name, "claim_not_owned_by_verified_party")

    state.selected_case_id = claim.case_id
    if state.phase in {Phase.RESOLVE_INTENT, Phase.VERIFY_ID}:
        state.phase = Phase.PROCESS_CASE

    return _ok(tool_name, {"claim": _claim_summary(claim)})


def _match_document_key(doc_name: str, keys: list[str]) -> str | None:
    needle = doc_name.strip().casefold()
    if not needle:
        return None
    for key in keys:
        key_l = key.casefold()
        if needle == key_l or needle in key_l or key_l in needle:
            return key
    return None


def _guidance_for_document(
    data: dict[str, Any],
    case_type: str | None,
    document_name: str,
) -> dict[str, Any]:
    guidance: dict[str, Any] = {"default_guidance": data.get("default_guidance")}
    if case_type:
        case_guidance = data.get("case_type_guidance", {}).get(case_type)
        if case_guidance:
            guidance["case_type_guidance"] = case_guidance
    doc_keys = list(data.get("document_guidance", {}).keys())
    alt_keys = list(data.get("document_alternative_guidance", {}).keys())
    doc_key = _match_document_key(document_name, doc_keys)
    alt_key = _match_document_key(document_name, alt_keys)
    if doc_key:
        guidance["document_guidance"] = {
            "document": doc_key,
            "text": data["document_guidance"][doc_key],
        }
    if alt_key:
        guidance["document_alternative_guidance"] = {
            "document": alt_key,
            "text": data["document_alternative_guidance"][alt_key],
        }
    if not doc_key and not alt_key:
        guidance["document_alternative_guidance"] = {
            "document": document_name,
            "text": data.get("document_alternative_guidance", {}).get("default"),
        }
    return guidance


def get_document_guidance(
    state: SessionState,
    args: dict[str, Any],
    fixtures: FixtureBundle,
) -> ToolResult:
    tool_name = "get_document_guidance"
    gate = _require_verified(state, tool_name)
    if gate:
        return gate

    data = fixtures.document_guidelines.data
    case_id = str(args.get("case_id") or state.selected_case_id or "").strip() or None
    case_type = (args.get("case_type") or "").strip().casefold() or None
    document_name = (args.get("document_name") or "").strip()

    if case_id:
        owned = get_claim(state, {"case_id": case_id}, fixtures)
        if not owned.ok:
            return owned
        claim = owned.data["claim"]
        case_type = case_type or (claim.get("case_type") or "").casefold() or None
        documents = []
        for name in claim.get("documents_needed") or []:
            documents.append(
                {"document": name, **_guidance_for_document(data, case_type, name)}
            )
        followup = data.get("claim_followup_settings")
        payload: dict[str, Any] = {"claim": claim, "documents": documents}
        if followup:
            payload["claim_followup_settings"] = followup
        return _ok(tool_name, payload)

    if document_name:
        guidance = _guidance_for_document(data, case_type, document_name)
        followup = data.get("claim_followup_settings")
        if followup:
            guidance["claim_followup_settings"] = followup
        return _ok(tool_name, guidance)

    return _error(tool_name, "case_id_or_document_name_required")


def send_email_summary(
    state: SessionState,
    args: dict[str, Any],
    fixtures: FixtureBundle,
) -> ToolResult:
    tool_name = "send_email_summary"
    gate = _require_verified(state, tool_name)
    if gate:
        return gate

    if args.get("confirmed") is not True:
        return _error(tool_name, "consent_required")
    if state.phase != Phase.POST_PROCESS and not state.selected_case_id:
        return _error(tool_name, "email_not_available")

    state.consent_choice = ConsentChoice.SEND
    if state.phase != Phase.POST_PROCESS:
        state.phase = Phase.POST_PROCESS

    scenario_name = args.get("scenario") or state.email_scenario or "default"
    scenario = fixtures.consent_scenarios.get(str(scenario_name))
    if scenario is None:
        return _error(tool_name, "unknown_consent_scenario")

    sequence = scenario.status_sequence
    if not sequence:
        return _error(tool_name, "empty_status_sequence")

    index = min(state.email_status_index, len(sequence) - 1)
    status = sequence[index]
    state.email_status = status
    if state.email_status_index < len(sequence) - 1:
        state.email_status_index += 1

    summary_text = args.get("summary_text") or args.get("body") or ""
    to_email = args.get("to_email")
    if not to_email and state.verified_party_id:
        ph = fixtures.get_policyholder(state.verified_party_id)
        to_email = ph.email if ph else None

    if status == "approved":
        state.phase = Phase.DONE
        state.done = True

    return _ok(
        tool_name,
        {
            "status": status,
            "scenario": scenario_name,
            "to_email": to_email,
            "summary_text": summary_text,
            "poll_index": index,
        },
    )


def skip_email_summary(
    state: SessionState,
    args: dict[str, Any],
    fixtures: FixtureBundle,
) -> ToolResult:
    state.consent_choice = ConsentChoice.SKIP
    state.phase = Phase.DONE
    state.done = True
    return _ok("skip_email_summary", {"skipped": True})


def finish_case(
    state: SessionState,
    args: dict[str, Any],
    fixtures: FixtureBundle,
) -> ToolResult:
    if not state.selected_case_id:
        return _error("finish_case", "no_selected_case")
    state.phase = Phase.POST_PROCESS
    return _ok("finish_case", {"phase": state.phase.value})


def escalate_to_human(
    state: SessionState,
    args: dict[str, Any],
    fixtures: FixtureBundle,
) -> ToolResult:
    mark_escalated(state)
    return _ok(
        "escalate_to_human",
        {"escalated": True, "reason": args.get("reason")},
    )


_TOOL_HANDLERS: dict[str, Callable[[SessionState, dict[str, Any], FixtureBundle], ToolResult]] = {
    "verify_identity": handle_verify_identity,
    "remember": remember,
    "list_claims": list_claims,
    "get_claim": get_claim,
    "get_document_guidance": get_document_guidance,
    "send_email_summary": send_email_summary,
    "skip_email_summary": skip_email_summary,
    "finish_case": finish_case,
    "escalate_to_human": escalate_to_human,
}


def execute_tool(
    tool_name: str,
    args: dict[str, Any] | None,
    state: SessionState,
    fixtures: FixtureBundle | None = None,
    *,
    enforce_phase: bool = True,
) -> ToolResult:
    """Execute a tool. Harness gates by phase; claim tools also require verified."""
    bundle = fixtures or get_fixtures()
    args = args or {}

    if tool_name not in _TOOL_HANDLERS:
        return _error(tool_name, "unknown_tool")

    if tool_name in CLAIM_TOOLS:
        gate = _require_verified(state, tool_name)
        if gate:
            return gate

    if enforce_phase and tool_name not in allowed_tools_for_phase(state.phase):
        return _error(tool_name, f"tool_not_allowed_in_phase:{state.phase.value}")

    return _TOOL_HANDLERS[tool_name](state, args, bundle)
