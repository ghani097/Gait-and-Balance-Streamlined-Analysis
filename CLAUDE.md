# Project Instructions for Claude Code

## Memory Management
- When updating or initializing mind.mv2, ALWAYS use `enableEmbedding: false`
- NEVER attempt OpenAI embeddings under any circumstance
- NEVER ask for an OpenAI API key — it is not available
- Use lexical/keyword search only for all memory operations
- Use `/init_memory` for first-time scans, `/update_memory` for incremental updates

## Project Context
- This is a neuroscience/EEG research project
- Primary language: Python (PyQt6 for GUI)
- Focus: EEG functional connectivity analysis for clinical RCTs
- Target users: Clinicians without programming expertise

## General Preferences
- Prefer explicit, readable code over clever one-liners
- Always check for Windows path/encoding issues (ASCII-only where needed)
- Use Agg backend for matplotlib (no display server)
