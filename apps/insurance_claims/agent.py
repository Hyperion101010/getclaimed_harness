from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from apps.insurance_claims.context import (
    TurnContext,
    claim_list_item,
    collected_pii_present,
    emit_reply,
    formatting_notes,
    identity_list_items,
    recent_transcript,
)
from apps.insurance_claims.escalation import ESCALATION_MESSAGE, mark_escalated, should_escalate
from apps.insurance_claims.guards import apply_output_guard
from apps.insurance_claims.llm import build_chat_model
from apps.insurance_claims.state import Phase, SessionState
from apps.insurance_claims.store import InMemorySessionStore, SessionStore
from apps.insurance_claims.tools import (
    ToolResult,
    execute_tool,
    langchain_tools_for_phase,
    missing_identity_fields,
)

MAX_TOOL_STEPS = 8

ScriptedCalls = Callable[[SessionState, str], list[tuple[str, dict[str, Any]]]]

AGENT_SYSTEM = """You are an insurance claims support agent.

You act by calling tools. You cannot set verified yourself. Only verify_identity can verify a caller.

Rules:
- During VERIFY_ID the only tools are verify_identity, remember, and escalate_to_human.
- Call remember when the caller mentions claim type, status, or month, even before identity is verified.
- Call verify_identity with any name, date of birth, phone, email, last four, and policy number from the utterance.
- Do not discuss claim files, case IDs, amounts, or denial reasons until identity is verified.
- After verification, list_claims with no arguments uses the stored hint. Pass all=true only if they ask for every claim.
- If list_claims returns count 0, say there are no matching claims. Do not invent IDs.
- If count is 1, you may call get_claim for that id.
- If several remain, list them and ask which one to open.
- get_document_guidance after a claim is selected, using that case_id.
- finish_case when the caller is done discussing the selected claim.
- send_email_summary only with confirmed=true after they clearly say yes or email me. Do not send on ok or silence.
- skip_email_summary if they decline the email.
- escalate_to_human if they insist on a person or verification keeps failing.

After you are done calling tools, write a short plain-text reply grounded only in this turn's tool results.
If still unverified, ask for remaining identity fields as a real bullet list, not a comma sentence.
"""


class TurnGraphState(BaseModel):
    session: SessionState
    user_message: str = ""
    reply: str = ""
    use_llm_reply: bool = True


def agent_system_prompt(session: SessionState) -> str:
    hint = session.case_hint.model_dump() if session.case_hint else None
    return (
        AGENT_SYSTEM
        + f"\nCurrent phase: {session.phase.value}\n"
        + f"Identity verified: {session.verified}\n"
        + f"Stored claim hint: {hint}\n"
        + f"Selected case: {session.selected_case_id}\n\n"
        + formatting_notes()
    )


def reply_from_tool_results(
    session: SessionState,
    results: list[ToolResult],
    user_message: str,
) -> str:
    if session.phase == Phase.ESCALATED:
        return ESCALATION_MESSAGE

    listed = next((item for item in reversed(results) if item.tool_name == "list_claims" and item.ok), None)
    got = next((item for item in reversed(results) if item.tool_name == "get_claim" and item.ok), None)
    docs = next(
        (item for item in reversed(results) if item.tool_name == "get_document_guidance" and item.ok),
        None,
    )
    emailed = next((item for item in reversed(results) if item.tool_name == "send_email_summary"), None)
    skipped = next(
        (item for item in reversed(results) if item.tool_name == "skip_email_summary" and item.ok),
        None,
    )

    list_items: list[str] = []
    list_style = None
    instruction = ""
    missing = missing_identity_fields(session)

    if not session.verified:
        instruction = (
            "Identity is not verified yet. Ask for identity details before any claim information. "
            "Use list_items as a bullet list of allowed fields. Do not write them as a comma sentence."
        )
        list_items = identity_list_items(missing)
        list_style = "bullet"
    elif skipped:
        instruction = "The caller declined the email. Confirm skip and close politely."
    elif emailed:
        if emailed.ok:
            status = (emailed.data or {}).get("status")
            if status == "approved":
                instruction = (
                    "The email tool reports approved. Confirm it was sent to the address in "
                    "tool_results and close."
                )
            else:
                instruction = (
                    "Consent was given and the email was submitted. Report the tool status as pending. "
                    "Do not claim it was delivered."
                )
        else:
            instruction = "The email was not sent. Ask whether to send or skip the email summary."
    elif listed and listed.data.get("count") == 0:
        instruction = (
            "There are no matching claims on file for this caller. Say that clearly. "
            "Do not invent claim IDs."
        )
    elif listed and int(listed.data.get("count") or 0) > 1:
        instruction = (
            "More than one claim still matches. List only the candidates in list_items "
            "and ask which one to open. Do not invent extra claims."
        )
        list_items = [claim_list_item(claim) for claim in listed.data.get("claims") or []]
        list_style = "numbered"
    elif got or session.selected_case_id:
        instruction = (
            "Answer using tool_results only: claim status, denial reason, documents needed, "
            "appeal deadline, amounts, and document guidance. Do not invent facts."
        )
        claim = (got.data.get("claim") if got else None) or {}
        if docs and docs.data.get("documents"):
            list_items = [item.get("document") for item in docs.data["documents"] if item.get("document")]
        else:
            list_items = list(claim.get("documents_needed") or [])
        if list_items:
            list_style = "bullet"
    else:
        instruction = (
            "Identity is verified but there is not enough of a claim hint yet. Ask whether "
            "this is healthcare, dental, or auto, and whether it is denied, open, or closed. "
            "Do not list claim IDs or statuses."
        )

    context = TurnContext(
        phase=session.phase,
        identity_verified=session.verified,
        instruction=instruction,
        user_message=user_message,
        sentiment=session.sentiment.value,
        missing_identity_fields=missing,
        collected_pii_present=collected_pii_present(session),
        case_hint=session.case_hint.model_dump() if session.case_hint else None,
        intent=session.intent.value if session.intent else None,
        selected_case_id=session.selected_case_id,
        tool_results=[item.model_dump() for item in results],
        recent_transcript=recent_transcript(session),
        list_style=list_style,
        list_items=list_items,
    )
    return emit_reply(context, use_llm=False)


def _tool_call_parts(call: Any) -> tuple[str, dict[str, Any], str]:
    if isinstance(call, dict):
        name = str(call.get("name") or "")
        args = call.get("args") or {}
        call_id = str(call.get("id") or name)
    else:
        name = str(getattr(call, "name", "") or "")
        args = getattr(call, "args", None) or {}
        call_id = str(getattr(call, "id", "") or name)
    if not isinstance(args, dict):
        args = {}
    return name, args, call_id


def _transcript_messages(session: SessionState) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in session.transcript:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    return messages


def run_llm_tool_loop(session: SessionState, user_message: str) -> tuple[str, list[ToolResult]]:
    results: list[ToolResult] = []
    try:
        llm = build_chat_model(temperature=0.2)
    except Exception:
        return reply_from_tool_results(session, results, user_message), results

    messages: list[BaseMessage] = [
        SystemMessage(content=agent_system_prompt(session)),
        *_transcript_messages(session),
    ]

    final_text = ""
    for _ in range(MAX_TOOL_STEPS):
        if session.phase in {Phase.DONE, Phase.ESCALATED}:
            break
        tools = langchain_tools_for_phase(session.phase)
        messages[0] = SystemMessage(content=agent_system_prompt(session))
        model = llm.bind_tools(tools) if tools else llm
        try:
            ai = model.invoke(messages)
        except Exception:
            return reply_from_tool_results(session, results, user_message), results
        messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            content = getattr(ai, "content", "") or ""
            if isinstance(content, list):
                final_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in content
                )
            else:
                final_text = str(content)
            break
        for call in tool_calls:
            name, args, call_id = _tool_call_parts(call)
            result = execute_tool(name, args, session)
            results.append(result)
            messages.append(
                ToolMessage(content=result.model_dump_json(), tool_call_id=call_id)
            )
        if session.phase == Phase.ESCALATED:
            return ESCALATION_MESSAGE, results

    return final_text.strip() or reply_from_tool_results(session, results, user_message), results


def run_scripted_tools(
    session: SessionState,
    user_message: str,
    scripted_calls: ScriptedCalls,
) -> list[ToolResult]:
    results: list[ToolResult] = []
    for name, args in scripted_calls(session, user_message):
        results.append(execute_tool(name, args, session))
        if session.phase == Phase.ESCALATED:
            break
    return results


def run_agent_turn(
    session: SessionState,
    user_message: str,
    *,
    use_llm_reply: bool = True,
    scripted_calls: ScriptedCalls | None = None,
) -> str:
    session.append_turn("user", user_message)

    if session.phase == Phase.ESCALATED:
        reply = ESCALATION_MESSAGE
        session.append_turn("assistant", reply)
        return reply
    if session.phase == Phase.DONE:
        reply = "This session is complete. Start a new session if you need more help."
        session.append_turn("assistant", reply)
        return reply

    if should_escalate(session):
        reply = mark_escalated(session)
        session.append_turn("assistant", reply)
        return reply

    results: list[ToolResult] = []
    if scripted_calls is not None:
        results = run_scripted_tools(session, user_message, scripted_calls)
        reply = reply_from_tool_results(session, results, user_message)
    elif use_llm_reply:
        reply, results = run_llm_tool_loop(session, user_message)
    else:
        reply = reply_from_tool_results(session, results, user_message)

    if should_escalate(session) and session.phase != Phase.ESCALATED:
        reply = mark_escalated(session)

    reply, _ = apply_output_guard(reply, session)
    session.append_turn("assistant", reply)
    session.touch()
    return reply


def build_turn_graph(
    *,
    use_llm_reply: bool = True,
    scripted_calls: ScriptedCalls | None = None,
):
    def turn_node(state: TurnGraphState) -> dict:
        reply = run_agent_turn(
            state.session,
            state.user_message,
            use_llm_reply=state.use_llm_reply if state.use_llm_reply is not None else use_llm_reply,
            scripted_calls=scripted_calls,
        )
        return {"session": state.session, "reply": reply}

    graph = StateGraph(TurnGraphState)
    graph.add_node("turn", turn_node)
    graph.set_entry_point("turn")
    graph.add_edge("turn", END)
    return graph.compile()


class ClaimsAgent:
    def __init__(
        self,
        store: SessionStore | None = None,
        *,
        use_llm_reply: bool = True,
        scripted_calls: ScriptedCalls | None = None,
    ) -> None:
        self.store = store or InMemorySessionStore()
        self.use_llm_reply = use_llm_reply
        self.scripted_calls = scripted_calls
        self.graph = build_turn_graph(
            use_llm_reply=use_llm_reply,
            scripted_calls=scripted_calls,
        )

    def start_session(self) -> SessionState:
        return self.store.create()

    def run_turn(self, session_id: str, user_message: str) -> tuple[SessionState, str]:
        session = self.store.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")

        result = self.graph.invoke(
            TurnGraphState(
                session=session,
                user_message=user_message,
                use_llm_reply=self.use_llm_reply,
            )
        )
        if isinstance(result, dict):
            final_state = TurnGraphState.model_validate(result)
        else:
            final_state = result

        saved = self.store.save(final_state.session)
        return saved, final_state.reply
