"""Verify every internal documentation link resolves. Run: python -m lottery.utils.check_docs

The documentation carries most of this project's reasoning, so a rotted
cross-reference is a real defect rather than cosmetic.

**Why this exists rather than an ad-hoc grep.** Headings here use `·` and `—`,
and GitHub's slugger *removes* those characters without collapsing the spaces
they leave behind — so `### 6 · Jugadas — generate, check, measure` anchors as
`6--jugadas--generate-check-measure`, with doubled hyphens. A checker that
normalises runs of hyphens looks correct and silently passes links that 404 on
GitHub; one such link sat broken in the docs precisely because the checker used
during review was that bit too forgiving. This reimplements github-slugger's
actual behaviour, including its `-1`, `-2` suffixes for duplicate headings.
"""

import glob
import os
import re
import sys

# The punctuation github-slugger strips. Characters are removed, not replaced,
# so the surrounding spaces survive and each becomes its own hyphen.
PUNCTUATION = re.compile(r"[ -⁯⸀-⹿\\'!\"#$%&()*+,./:;<=>?@\[\]^`{|}~·]")
HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.M)
LINK = re.compile(r"\]\(([^)]+)\)")


def slugify(heading):
    return PUNCTUATION.sub("", heading.strip().lower()).replace(" ", "-")


def anchors_in(path):
    """Every anchor a Markdown file defines, with GitHub's duplicate suffixes."""
    seen, anchors = {}, set()
    with open(path) as handle:
        for match in HEADING.finditer(handle.read()):
            slug = slugify(match.group(1))
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def broken_links(files):
    """Every (file, link, nearest candidates) whose target does not exist."""
    anchors = {path: anchors_in(path) for path in files}
    problems = []
    for path in files:
        with open(path) as handle:
            content = handle.read()
        for match in LINK.finditer(content):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, _, anchor = target.partition("#")
            base = os.path.normpath(os.path.join(os.path.dirname(path), file_part)) if file_part else path
            if file_part and not os.path.exists(base):
                problems.append((path, target, ["file does not exist"]))
            elif anchor and base in anchors and anchor not in anchors[base]:
                stem = anchor.split("-")[0]
                problems.append((path, target, sorted(a for a in anchors[base] if stem in a)))
    return problems


def markdown_files(root="."):
    return sorted(glob.glob(os.path.join(root, "docs", "*.md"))) + [
        os.path.join(root, name) for name in ("README.md", "CLAUDE.md")
        if os.path.exists(os.path.join(root, name))
    ]


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    files = markdown_files(REPO_ROOT)
    problems = broken_links(files)
    for path, target, candidates in problems:
        print(f"{path}: broken link -> {target}")
        print(f"    did you mean: {candidates}" if candidates else "    no similar anchor found")

    if problems:
        print(f"\n{len(problems)} broken link(s) across {len(files)} files.")
        raise SystemExit(1)
    print(f"All internal links resolve across {len(files)} files.")
