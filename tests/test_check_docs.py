"""The documentation link checker — lottery/utils/check_docs.py.

This module exists because of one near-miss: headings here use `·` and `—`, and
GitHub's slugger *removes* those characters without collapsing the spaces they
leave behind, so `### 6 · Jugadas — generate, check, measure` anchors as
`6--jugadas--generate-check-measure` with doubled hyphens. A checker that
normalises runs of hyphens reports success on links that 404 in the browser, and
one broken cross-reference survived several review passes for exactly that
reason.

So the load-bearing test here is not "does it find broken links" but **does it
keep the doubled hyphens**. A future tidy-up of `slugify` that collapses them
would leave every test about broken links passing and silently reintroduce the
bug the module was written to catch.
"""

import pytest

from lottery.utils.check_docs import (
    REPO_ROOT,
    anchors_in,
    broken_links,
    markdown_files,
    slugify,
)

# --- slugify: github-slugger's behaviour, not an approximation of it ----------

def test_the_characters_github_strips_leave_their_spaces_behind():
    # The whole reason this module is not a grep. Two hyphens, not one.
    assert slugify("6 · Jugadas — generate, check, measure") == "6--jugadas--generate-check-measure"


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("Setup and commands", "setup-and-commands"),
        ("Two time axes", "two-time-axes"),
        ("What this project is", "what-this-project-is"),
        ("3.5 Continuous integration", "35-continuous-integration"),   # the dot is removed
        ("Odds are not probabilities.", "odds-are-not-probabilities"),
        ("`beats_chance_test`", "beats_chance_test"),                  # underscores survive
        ("MDE (minimum detectable effect)", "mde-minimum-detectable-effect"),
    ],
)
def test_a_heading_slugs_the_way_github_slugs_it(heading, expected):
    assert slugify(heading) == expected


def test_leading_and_trailing_whitespace_does_not_reach_the_slug():
    assert slugify("  Setup and commands  ") == "setup-and-commands"


# --- anchors_in: duplicate headings get GitHub's -1, -2 suffixes -------------

def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_every_heading_level_defines_an_anchor(tmp_path):
    path = write(tmp_path, "a.md", "# Top\n\n### Deep one\n\n###### Deepest\n")

    assert anchors_in(path) == {"top", "deep-one", "deepest"}


def test_repeated_headings_get_the_numbered_suffixes_github_gives_them(tmp_path):
    path = write(tmp_path, "a.md", "## Notes\n\n## Notes\n\n## Notes\n")

    # The first keeps the bare slug; only the later ones are suffixed.
    assert anchors_in(path) == {"notes", "notes-1", "notes-2"}


# --- broken_links -------------------------------------------------------------

def test_a_link_to_an_anchor_that_exists_is_not_reported(tmp_path):
    path = write(tmp_path, "a.md", "# Top\n\nSee [the top](#top).\n")

    assert broken_links([path]) == []


def test_a_link_to_a_missing_anchor_is_reported(tmp_path):
    path = write(tmp_path, "a.md", "# Top\n\nSee [nowhere](#bottom).\n")

    problems = broken_links([path])

    assert [(p, target) for p, target, _ in problems] == [(path, "#bottom")]


def test_a_near_miss_anchor_is_offered_as_a_candidate(tmp_path):
    """Naming the near miss is the difference between a report and a scavenger hunt."""
    path = write(tmp_path, "a.md", "## Setup and commands\n\n[link](#setup-and-flags)\n")

    _, _, candidates = broken_links([path])[0]

    assert "setup-and-commands" in candidates


def test_a_link_to_a_file_that_does_not_exist_is_reported(tmp_path):
    path = write(tmp_path, "a.md", "See [the other page](missing.md).\n")

    _, target, candidates = broken_links([path])[0]

    assert target == "missing.md"
    assert candidates == ["file does not exist"]


def test_a_cross_file_anchor_resolves_against_the_file_it_names(tmp_path):
    other = write(tmp_path, "b.md", "## The premise\n")
    path = write(tmp_path, "a.md", "[good](b.md#the-premise) and [bad](b.md#the-premises)\n")

    problems = broken_links([path, other])

    assert [target for _, target, _ in problems] == ["b.md#the-premises"]


def test_external_links_are_left_alone(tmp_path):
    path = write(
        tmp_path,
        "a.md",
        "[http](http://example.com/x.md) [https](https://example.com#nope) [mail](mailto:a@b.c)\n",
    )

    assert broken_links([path]) == []


def test_a_doubled_hyphen_anchor_resolves_rather_than_being_reported(tmp_path):
    """The regression this module exists for, end to end."""
    path = write(
        tmp_path,
        "a.md",
        "### 6 · Jugadas — generate, check, measure\n\n"
        "[jump](#6--jugadas--generate-check-measure)\n",
    )

    assert broken_links([path]) == []


# --- the repository's own documentation --------------------------------------

def test_markdown_files_finds_the_docs_directory_and_the_two_root_pages():
    files = markdown_files(REPO_ROOT)

    names = {f.rsplit("/", 1)[-1] for f in files}
    assert {"README.md", "CLAUDE.md", "development.md"} <= names


def test_this_repository_s_own_internal_links_all_resolve():
    """The same check the `static` CI job runs, so a rotted link fails pytest too."""
    problems = broken_links(markdown_files(REPO_ROOT))

    assert problems == [], "\n".join(f"{path}: {target} (near: {near})" for path, target, near in problems)
