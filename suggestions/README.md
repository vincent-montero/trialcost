# Reviewer suggestions

Reviewers edit values in the app (Assumptions & suggestions tab) and download a
`suggestions-YYYYMMDD-<name>.csv`. Nothing they do reaches `data/` (D40).

1. Save received files in `suggestions/inbox/` (git-ignored: files may contain reviewer names).
2. Review: `uv run python -m trialcost.suggestions review suggestions/inbox/<file>.csv [--design designs/<x>.yaml]`
   shows each suggestion, any validation problem, and its impact on the design's mean total cost.
3. Apply the accepted ones: `uv run python -m trialcost.suggestions apply suggestions/inbox/<file>.csv --accept 1,3 [--grade C]`
   edits only those cells in `data/*.csv`, sets the grade, adds "Expert review YYYY-MM-DD" to the
   source and the rationale to the notes, re-validates the catalogue (files restored on failure).
4. Run the tests, update README §6 and the work log, commit.
