# Initialize Project Memory

Perform a full first-time memory scan of this codebase and store everything in mind.mv2.

## CRITICAL RULES - READ FIRST
- ALWAYS use `enableEmbedding: false` — no exceptions
- NEVER attempt OpenAI embeddings
- NEVER ask for an OpenAI API key
- Use lexical/keyword search only throughout

## What to Scan and Store

### 1. Project Overview (type: discovery)
- What this project does
- Its purpose and goals
- Who uses it (clinicians, researchers, etc.)
- Tech stack (Python, PyQt6, MATLAB, etc.)

### 2. File Structure (type: discovery)
- Top-level folders and their roles
- Main entry point(s)
- Key modules and what they do
- Config files and their purpose

### 3. Architecture Patterns (type: pattern)
- GUI architecture (if PyQt6 — workers, signals, slots)
- Pipeline flow (how data moves through the system)
- Key classes and their responsibilities
- Important functions and algorithms

### 4. EEG/Neuroscience Specifics (type: pattern)
- EEG preprocessing steps used
- Artifact removal methods (ASR, ICLabel, GEDAI, etc.)
- Frequency bands configured
- Connectivity measures used (PLI, coherence, etc.)
- Source localization methods (sLORETA, etc.)

### 5. Dependencies & Environment (type: discovery)
- Python packages used
- Any MATLAB dependencies
- Virtual environment name/path
- Platform quirks (Windows-specific, encoding issues, etc.)

### 6. Known Issues & Warnings (type: warning)
- Known bugs or limitations
- Platform-specific workarounds
- Any TODOs or incomplete features

### 7. Recent Changes (type: discovery)
- What was most recently worked on
- Current state of development
- What is working vs what is in progress

## Memory Storage Instructions
- Store each category as a separate memory entry
- Use descriptive summaries (2-5 sentences each)
- Tag appropriately: discovery, pattern, warning
- Set `enableEmbedding: false` on every single store call
- Confirm total memories written at the end
