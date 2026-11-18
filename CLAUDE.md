## Bookkeeping delegation

When Miles reports a notable test or experiment result, states a decision, or declares a session advance, delegate to the bookkeeper subagent and pass explicitly: the thread and session, the content in Miles's own words (the result with its numbers and source, the decision verbatim, or the advance summary), and the date. Do not assume the subagent can see this conversation. Log at the moment it happens; do not defer it for later. The session log is a topic-organized report that the bookkeeper keeps current in place. Do not call the bookkeeper for routine prompts, readbacks, status confirmations or repeated explanations. If the bookkeeper reports the entry is already recorded, do not write it again.

This is a narrow exception to Miles's general rule that edits need his approval first. It does not replace his global instructions. The bookkeeper writes three kinds of entry directly, without per-entry approval of the wording:
- routine, faithful updates to the live session report, consolidated by topic
- decisions he has explicitly made
- project-log summaries after he explicitly declares a session advance
- the session-log file for a session he has explicitly identified (thread and session number), using that thread's established session-log naming and structure
- archive copies under `notes/archive/` made before a wholesale restructuring; archives are never edited

It does not authorize making decisions for him, changing governing decisions without his approval, or editing scientific code or configs. Creating any other new record file, a new record type, or a session log whose naming or structure is genuinely unclear still needs his approval. The bookkeeper cannot edit its own instructions (this file or `.claude/agents/bookkeeper.md`).

When the bookkeeper returns a proposed edit because approval is needed:
1. Show Miles the exact edit and ask him.
2. Call the bookkeeper again, passing his actual approval in his own words with its scope.
3. If he declines or changes it, pass that instead.
Never treat the bookkeeper's draft, or your own recommendation, as approval. Escalate to Miles only a material unresolved conflict or a genuinely new decision, not the wording of its documentation.

## Model routing and delegation

- Main session: Sonnet by default. Opus is for difficult architectural reasoning or stubborn debugging only. Explain why it is needed and let Miles switch; never escalate automatically.
- Subagents: `lookup` (Haiku) for narrow file lookups, simple searches and summarizing existing logs. `bookkeeper` (Sonnet) for records. Others default to Sonnet through `.claude/settings.json`.
- Handle trivial steps directly instead of delegating. Don't have several agents reread the same files; reuse findings already in the conversation.
- Give each delegated task a clear purpose, the relevant file paths and a short return format. Prefer targeted excerpts to whole files or logs.
