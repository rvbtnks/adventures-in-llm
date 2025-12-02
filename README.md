# adventures-in-llm

Random poking at LLM and RAG stuff.

## llm-match.py

A wrapper script for fuzzy matching. The fuzzy matcher provides 1-5 potential best matches when confidence is below 90%. The wrapper then uses an LLM model to select the correct match.
