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
Pokemon names with gender symbols (♀, ♂) in data dictionaries are acceptable since they're game data, not code logic.

