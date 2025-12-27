# Coding Rules for GBC Emulator

## NO UNICODE CHARACTERS - ASCII ONLY

**CRITICAL RULE**: Never use Unicode characters (emojis, special symbols) in code, strings, or output.

### Why?
- Windows console uses CP1252 encoding which doesn't support Unicode
- Causes `UnicodeEncodeError` crashes
- Ensures cross-platform compatibility

### ASCII Alternatives

| Unicode | ASCII Replacement |
|---------|-------------------|
| → | `->` or `>>` |
| ✓ | `[OK]` |
| ✗ | `[X]` |
| 🎉 | `***` |
| ⚡ | `[FAST]` |
| 🎯 | `[GOAL]` |
| 📍 | `[LOC]` |
| ⚔️ | `[BATTLE]` |
| 💭 | `[AI]` |
| ❌ | `[X]` |

### Examples

**BAD:**
```python
self._log(f"✓ GOAL COMPLETE: {reason}")
print(f"🎉 Success!")
label = f"→ {option}"
```

**GOOD:**
```python
self._log(f"[OK] GOAL COMPLETE: {reason}")
print(f"*** Success! ***")
label = f">> {option}"
```

### Exception
Pokemon names with gender symbols in data dictionaries are acceptable since they're game data, not code logic.

---

## DO NOT DISABLE LLM

**CRITICAL RULE**: Never disable the LLM/Ollama integration for performance reasons.

### Why?
- The AI agent's intelligence depends on LLM for complex decisions
- Async LLM calls don't block the game loop
- Heuristics alone are too simple for proper gameplay
- User expects LLM-powered decision making

### What to do instead
- Use async/threaded LLM calls (already implemented)
- Increase frame_skip to reduce decision frequency
- Use time-based caching to reduce state reads
- Optimize other areas (memory reads, rendering)

### The rule
```python
# CORRECT - LLM enabled
self.fast_mode = False  # Keep LLM enabled

# WRONG - Do not do this
self.fast_mode = True  # Disable LLM
```

