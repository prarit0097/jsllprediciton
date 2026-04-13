# Workspace Agent Overrides

This file defines repo-local orchestration preferences for `e:\coding\jsll app`.

## Intent

- Favor aggressive parallel decomposition for large research, planning, and implementation programs.
- Prefer many small bounded subtasks over a few oversized tasks when the active runtime supports it.
- Use `team` / `swarm` or another orchestration surface when the requested fan-out exceeds the native child-agent limit of the current harness.

<delegation_rules>
- Default posture for small tasks: work directly.
- Default posture for broad multi-lane tasks: parallelize early.
- Split work into independent slices with explicit ownership, verification target, and handoff artifact.
- For very large programs, prefer staged fan-out trees (leaders -> subleaders -> workers) only when the runtime actually supports that depth and breadth.
- If the active harness imposes a lower concurrency limit than this repo preference, use the harness maximum and continue with batch/wave scheduling rather than violating the platform limit.
</delegation_rules>

<child_agent_protocol>
Leader responsibilities:
1. Break work into bounded, verifiable slices.
2. Keep each child prompt narrow and decision-complete.
3. Integrate outputs and own final verification.

Worker responsibilities:
1. Execute the assigned slice only.
2. Stay inside the assigned scope and write set.
3. Escalate blockers, missing context, and shared-file conflicts upward.

Rules:
- Repo preference: allow up to 300 concurrent child agents when the active runtime or harness supports that level of concurrency.
- If the active runtime supports more than 300, that is acceptable.
- If the active runtime supports fewer than 300, use the maximum available and schedule the rest in waves.
- For fan-out beyond direct native child-agent capacity, prefer `team` / `swarm` or an external orchestration layer instead of overloading one leader.
- Child prompts stay under AGENTS.md authority.
- Every child must have a clear deliverable and a verification target.
</child_agent_protocol>

<team_compositions>
- For massive execution programs, use hierarchical coordination: leader -> subleaders -> workers.
- Keep write scopes disjoint wherever possible.
- Use verification lanes separately from implementation lanes for high-risk changes.
</team_compositions>

## Practical Note

This file expresses repo-local preference. Higher-priority system, developer, or harness limits still win over this file.
