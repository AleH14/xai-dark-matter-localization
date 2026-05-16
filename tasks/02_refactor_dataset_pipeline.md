# Fix notebook cell types and execution order

## Goal
Fix notebooks so they can be executed from top to bottom without syntax errors caused by markdown text inside code cells.

## Files to inspect
- notebooks/02_select_subhalos.ipynb
- all notebooks in notebooks/

## Requirements
- Convert any explanatory text cells incorrectly marked as code into Markdown cells.
- Do not change scientific logic.
- Ensure notebook headings and descriptions are clear.
- Add a short "Expected output" markdown cell at the end of each notebook.

## Acceptance criteria
- notebooks/02_select_subhalos.ipynb no longer has markdown text inside code cells.
- All notebooks can be opened and run sequentially.