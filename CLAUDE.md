## Bookkeeping delegation

When Miles reports a notable test or experiment result, states a decision, or declares a session advance, delegate to the bookkeeper subagent and pass explicitly: the thread and session, the content (the result with its numbers and source, the decision and its scope, or the advance summary), and the date. Records never quote Miles verbatim. Do not assume the subagent can see this conversation. Log at the moment it happens; do not defer it for later. A missing log entry does not make a known result unknown. Flag missed documentation to Miles for his decision and preserve existing records. Once the missing result or original entry is recovered and the later entry is authorized, replace the missing-result flag in place with the recovered information, so one clear, complete entry remains. Mark it "previously missing", state when and where the information was recovered, and give the original event date separately when known. Do not invent missing details or rewrite unrelated history. The session log is a topic-organized report at the scale of `notes/Tripwire_CYP_004.md` that the bookkeeper keeps current in place; exhaustive hashes and check-by-check evidence go in a separate supporting record. Do not call the bookkeeper for routine prompts, readbacks, status confirmations or repeated explanations. If the bookkeeper reports the entry is already recorded, do not write it again.

This is a narrow exception to Miles's general rule that edits need his approval first. It does not replace his global instructions. The bookkeeper writes three kinds of entry directly, without per-entry approval of the wording:
- routine, faithful updates to the live session report, consolidated by topic
- decisions he has explicitly made
- project-log summaries after he explicitly declares a session advance
- the session-log file for a session he has explicitly identified (thread and session number), using that thread's established session-log naming and structure
- archive copies under `notes/archive/` made before a wholesale restructuring; archives are never edited

It does not authorize making decisions for him, changing governing decisions without his approval, or editing scientific code or configs. Creating any other new record file, a new record type, or a session log whose naming or structure is genuinely unclear still needs his approval. The bookkeeper cannot edit its own instructions (this file or `.claude/agents/bookkeeper.md`).

When the bookkeeper returns a proposed edit because approval is needed:
1. Show Miles the exact edit and ask him.
2. Call the bookkeeper again, passing his actual approval and its scope.
3. If he declines or changes it, pass that instead.
Never treat the bookkeeper's draft, or your own recommendation, as approval. Escalate to Miles only a material unresolved conflict or a genuinely new decision, not the wording of its documentation.

## Model routing and delegation

- Main session: Sonnet by default. Opus is for difficult architectural reasoning or stubborn debugging only. Explain why it is needed and let Miles switch; never escalate automatically.
- Subagents: `lookup` (Haiku) for narrow file lookups, simple searches and summarizing existing logs. `bookkeeper` (Sonnet) for records. Others default to Sonnet through `.claude/settings.json`.
- Handle trivial steps directly instead of delegating. Don't have several agents reread the same files; reuse findings already in the conversation.
- Give each delegated task a clear purpose, the relevant file paths and a short return format. Prefer targeted excerpts to whole files or logs.

## Implementation workflow

Proceed with code inspection, reasoning, record searches and preparation of proposed edits without asking permission for each step. Do not stop after every read to propose another read.

For unresolved implementation choices, finish the investigation and present one consolidated plan with a recommended design and reasons. Miles approves the plan once. That approval covers the stated code/config edits and new files; do not request approval again for each file. Pause only for a material scope change, a genuinely new scientific decision or a conflict that the existing evidence cannot resolve.

Recover settled decisions and previously supplied paths/results from the conversation and project records. Missing documentation is not automatically an undecided question. Record clear existing decisions faithfully; do not invent a resolution when the evidence conflicts.

Before asking for missing inputs, recover everything available locally and from earlier outputs. Consolidate what remains into one targeted request. When CoreHPC access is needed, provide exact read-only lookup commands using known locations, without guessed paths or flags. Do not repeat a lookup whose output is already available unless there is a specific reason it may have changed.

Miles runs execution commands, tests, transfers and SLURM submissions himself. Prepare the commands as a coherent sequence with the machine, directory, purpose and stopping conditions identified. Missing permission for Claude itself to execute is not an implementation blocker. Code-edit approval does not authorize execution.

Tie each proposed check to a concrete failure risk. Explain what existing evidence already covers and what remains untested. A changed hash identifies changed evidence scope; it does not by itself justify another GPU job. Honor explicit project-specific waivers without extending them to unrelated checks.

Report completed work, meaningful findings and actual blockers concisely. Avoid repeated status readbacks, per-file approval loops and serial fragments of a plan that could have been presented together.
