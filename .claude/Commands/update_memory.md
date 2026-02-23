# Update Project Memory

Scan this codebase for changes and update mind.mv2 with anything new or modified.

## CRITICAL RULES - READ FIRST
- ALWAYS use `enableEmbedding: false` — no exceptions
- NEVER attempt OpenAI embeddings
- NEVER ask for an OpenAI API key
- Use lexical/keyword search only throughout

## What to Look For

### 1. New Files
- Any files not previously in memory
- New modules, scripts, or configs added

### 2. Modified Patterns
- Functions or classes that have changed significantly
- Pipeline steps that were refactored
- New algorithms or processing methods added

### 3. Resolved Issues
- Bugs that have been fixed since last scan
- Workarounds that are no longer needed
- Remove or update any outdated warning memories

### 4. New Decisions
- Architectural choices made recently
- New dependencies added
- Configuration changes

### 5. Current State Update
- Update the "current state of development" memory
- What is now working that wasn't before
- What is currently in progress

## Memory Update Instructions
- Only add memories for things that are NEW or CHANGED
- Do not duplicate memories that already exist
- Use descriptive summaries (2-5 sentences each)
- Tag appropriately: discovery, pattern, warning
- Set `enableEmbedding: false` on every single store call
- Confirm how many memories were added/updated at the end
