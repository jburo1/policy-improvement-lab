# AGENTS.md

General instructions for coding agents. Follow applicable repository-specific instructions in addition to this file.

## Prime Directive

**Solve the actual problem with the least necessary complexity.**

- Deliver a correct, complete solution to the current requirements—not imagined future ones.
- Prefer the simplest design that preserves necessary correctness, safety, reliability, and maintainability.
- Judge simplicity by the system's overall conceptual and maintenance burden, not by line count or diff size.
- Avoid speculative features, premature generalization, unnecessary abstraction, and infrastructure without a present need.
- Reduce complexity when doing so is part of solving the task; do not use a focused task as an excuse for unrelated cleanup.

## Scope and Judgment

- Fulfill the user's objective and explicit constraints. Do not silently redefine the task.
- If a requested approach adds substantial unnecessary complexity, briefly identify a simpler alternative. Respect an explicit decision once made.
- Prefer the smallest coherent change that fully solves the problem. Expand scope only when correctness requires it.
- Preserve existing behavior and public contracts unless the task calls for changing them.
- Follow established repository conventions unless there is a concrete reason not to.
- Report important adjacent problems rather than fixing them opportunistically.

## Working Method

```text
inspect -> implement -> verify -> report
```

1. Inspect the relevant implementation, tests, nearby documentation, and applicable nested `AGENTS.md` before editing. Read enough to understand the affected boundary, not the entire repository by default.
2. Distinguish observed behavior from assumptions and hypotheses. Investigate uncertainty that could change the solution.
3. For nontrivial work, identify the simplest viable approach before coding. Skip ceremonial planning for obvious changes.
4. Implement directly and keep the diff focused. Review the resulting diff for accidental or unrelated edits.
5. Verify at the narrowest meaningful boundary, then report what the evidence does and does not establish.

## Design and Implementation

- Prefer clear, explicit code and straightforward data flow over cleverness.
- Reuse existing mechanisms before introducing another layer, framework, or dependency.
- Introduce abstractions when they remove demonstrated complexity, not merely to anticipate reuse.
- Avoid configuration options, compatibility layers, fallbacks, and extension points without a concrete requirement.
- Keep failures visible: do not hide errors behind broad exception handling, silent defaults, or misleading success states.
- Respect relevant constraints on performance, security, concurrency, resource use, and operational behavior.
- Write comments for non-obvious intent or constraints; avoid comments that merely restate the code.
- Update documentation when the change alters documented behavior, interfaces, or operational procedures—not as a ritual.

## Verification and Evidence

- Choose checks proportionate to the change's risk and scope: targeted tests first, broader checks when needed.
- For defects, reproduce the failure when practical and verify the fix addresses it.
- Add or update focused tests when they meaningfully protect the changed behavior.
- Do not conflate passing unit tests, compilation, simulation, integration testing, and production or hardware validation.
- Measure performance claims under relevant conditions when feasible; otherwise label them as expectations.
- Never claim checks passed unless they were actually run. State untested boundaries and blockers plainly.

## Safety and Repository Hygiene

- Preserve unrelated user changes. Do not reset, overwrite, or delete work outside the assigned scope.
- Do not perform destructive operations or change production systems, credentials, persistent data, or deployment targets without explicit authorization.
- Do not commit, push, publish, deploy, or trigger remote CI/CD unless asked.
- Do not introduce secrets, machine-specific paths, or unnecessary generated artifacts into version control.
- Treat migrations, external side effects, and irreversible actions as requiring deliberate review.

## Communication

- Keep explanations and final reports concise and proportional to the task.
- State what changed, what was verified, and any material limitations or unresolved issues.
- Separate facts established by inspection or testing from inferences and recommendations.
- Prefer a direct solution and clear evidence over elaborate plans or lengthy self-justification.

## Scientific Correctness

- Preserve the mathematical meaning of implemented algorithms,
  objectives, and measurements. Do not silently change learning
  or experimental semantics.
- Use exact MDP values only for evaluation and diagnostics,
  never to guide learning unless explicitly implementing
  an oracle control.
- Make experimental comparisons controlled and interpretable.
  Preserve the information needed to identify training runs,
  checkpoints, and compared methods.
- Freeze policy and critic states for paired comparisons; state
  whether checkpoint measurements precede or follow learning updates.
- Test important mathematical properties directly rather than
  relying only on training outcomes or plausible plots.
- Distinguish observed results from hypotheses and interpretations.
  Never modify algorithms or experimental data merely to
  produce expected results.

## Version control
- When explicitly instructed to commit changes, use the Conventional
  Commits format: `type(scope): concise imperative description`.
  Common types include `feat`, `fix`, `refactor`, `test`, `docs`,
  `chore`, and `ci`. Omit the scope when unnecessary. Keep commit
  messages accurate and focused on the changes made.
