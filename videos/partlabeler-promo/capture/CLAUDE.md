# PartLabeler

Source: http://127.0.0.1:8766/

To create a video from this capture, use the `product-launch-video` skill.

## What's in This Capture

| File | Contents |
|------|----------|
| `screenshots/contact-sheet.jpg` | **View this first.** All scroll screenshots in labeled grid — see the entire page at a glance |
| `screenshots/scroll-*.png` | Individual viewport screenshots if you need detail on a specific section. |
| `extracted/tokens.json` | Design tokens: 12 colors, 0 fonts, 6 headings, 3 CTAs |
| `extracted/design-styles.json` | Computed styles from live DOM: typography hierarchy, button/card/nav styles, spacing scale, border-radius, box shadows. Primary data source for DESIGN.md. |
| `extracted/asset-descriptions.md` | One-line description of every downloaded asset. Read this for asset selection — only open individual files for safe-zone checking. |
| `extracted/visible-text.txt` | Page text in DOM order, prefixed with HTML tag (`[h1]`, `[p]`, `[a]`). Use as context — rephrase freely. |
| `assets/contact-sheet.jpg` | All downloaded images in one labeled grid. |
| `assets/` | Individual downloaded images, SVGs, and font files. |

## Brand Summary

- **Colors**: #15202B (accent), #FFFFFF (bg-light), #F3F5F7 (bg-light), #D7DDE3 (surface-light), #0A7C78 (accent), #5B6874 (neutral), #E4E9ED (bg-light), #2F9E5B (accent), #6AA6F2 (accent), #E1F0EF (bg-light)
- **Fonts**: none detected
