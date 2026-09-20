[English](maintenance.md) · [简体中文](maintenance.zh-CN.md)


# Maintenance and handoff

Julius maintains the project direction, reviews proposed tasks and coordinates releases. Contribution records credit the people who actually designed, implemented, executed or reviewed each artifact. Named co-maintainers can be added after accepting a defined responsibility.

| Responsibility | Required record |
| --- | --- |
| Task triage | Issue, scope, owner, acceptance criterion and dependencies |
| Technical review | Source/version, assumptions, checks and unresolved limitations |
| Documentation | English/Chinese parity, working links, manuscript references, readable original figures and asset hashes |
| Release | Reviewed file list, license/attribution, validation record and commit |
| Handoff | Last working entry point, environment, next action and decision needing attention |

Before handing off a task, provide a minimal reading order and explain what has actually been checked. Keep credentials in the execution environment. Record where an authorized collaborator obtains access without placing access details in a public document.

When a result changes, update its source table, both languages and the relevant case study together. Keep historical results associated with their own versions. When maintenance pauses, record the last supported snapshot and open tasks so another contributor can make a bounded improvement.

For a manuscript or figure review, retain the section/table/figure number and identify the evidence as historical design, manuscript-reported result, current source behavior, or observed execution. Compare asset bytes with [provenance.json](assets/manuscript/provenance.json) and review every English/Chinese caption that uses the image. The eight-dimensional manuscript state and nine-field current implementation must remain distinguishable.

A useful handoff reading order is the README, the relevant case, its linked manuscript passage, then the exact source symbol. State the next review question and its acceptance check so a contributor can continue without reconstructing earlier discussion.

The [contribution guide](../CONTRIBUTING.md) contains the record template.
