# TradeGenius Design System v1.0

> Bloomberg clarity + Robinhood simplicity + Apple spacing

This file is the **single source of truth** for all visual decisions in TradeGenius.

**Before making any UI change, read this file.**
**Before writing any color or font-size, check this file.**
**If a value is not in this file, it is not allowed.**

---

## Colors

### Backgrounds
| Token    | Value     | Use |
|----------|-----------|-----|
| BG       | #0f1115   | Page background |
| SURFACE  | #171a21   | Card background |
| SURFACE2 | #222733   | Elevated / hover |
| BORDER   | #2d3445   | Borders, dividers |

### Text — exactly 3 levels
| Token | Value   | Use |
|-------|---------|-----|
| TEXT1 | #ffffff | All values, numbers, amounts |
| TEXT2 | #b0b7c3 | Labels, captions, timestamps |
| TEXT3 | #7f8896 | Helper text, placeholders |

### Actions — consistent everywhere
| Action | Color   | Background |
|--------|---------|------------|
| BUY    | #00c853 | #00200d    |
| ADD    | #00c853 | #00200d    |
| SELL   | #ff5252 | #200808    |
| EXIT   | #ff5252 | #200808    |
| TRIM   | #ffb300 | #1f1500    |
| HOLD   | #64b5f6 | #081428    |
| WATCH  | #ab47bc | #150820    |

## Typography — exactly 4 sizes
| Token        | Size | Use |
|--------------|------|-----|
| FONT_HERO    | 36px | Portfolio value, health score |
| FONT_SECTION | 20px | Card titles |
| FONT_VALUE   | 15px | Data values, prices, % |
| FONT_LABEL   | 11px | Labels (uppercase only) |

**Any other font size is a design system violation.**

## Spacing
| Token        | Value  | Use |
|--------------|--------|-----|
| CARD_PADDING | 20px   | All card inner padding |
| CARD_RADIUS  | 12px   | All card border radius |
| ROW_PADDING  | 12px 0 | Table and list rows |
| SECTION_GAP  | 16px   | Between sections |

## Component Rules

### Cards
- Always use `_card()` wrapper
- Minimum padding: 20px
- Border radius: 12px
- Hero cards get `accent_color` top border

### Badges
- Always use `_action_badge()`
- Colors **never** overridden
- EXIT and SELL: `large` size
- HOLD: `small` size

### Symbols
- Always use `_symbol()`
- Always monospace
- Always ACTION_BUY green
- Always font-weight 700

### Confidence
- Always use `_confidence_bar()`
- Always show BOTH number and bar
- Never show only one

### Tables
- Columns: **Symbol | Action | Weight | Target | Confidence | P&L**
- No accounting columns (shares, cost basis, invested)
- Always use `_table()` wrapper
- Always `overflow-x:auto` for mobile

## Action Visual Hierarchy
| Action | Color  | Badge Size | Row Style |
|--------|--------|------------|-----------|
| EXIT   | Red    | Large      | Full row red highlight |
| SELL   | Red    | Large      | Full row red highlight |
| TRIM   | Amber  | Large      | Amber left border |
| BUY    | Green  | Large      | Green left border |
| ADD    | Green  | Normal     | Green left border |
| WATCH  | Purple | Normal     | No highlight |
| HOLD   | Blue   | Small      | Dimmed row |

## Mobile Rules
- Must work at 390px width
- No horizontal scrolling
- All widths in % not px
- `flex-wrap:wrap` on all row elements

## What is NOT allowed
- Font sizes other than 36/20/15/11px
- Text colors other than #ffffff/#b0b7c3/#7f8896
- Action colors other than the 5 defined above
- Pure black (#000000) anywhere
- Fixed pixel widths over 300px
- More than 3 text colors in any panel
- Confidence shown without both number and bar
- Any position shown without an action badge
- Old color values: #0e0e0e, #1b1b1b, #a0a0a0, #00c805, #ff5000, #9d4edd
