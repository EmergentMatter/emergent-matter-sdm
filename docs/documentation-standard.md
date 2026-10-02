# The documentation standard

Code describes itself. A second description of the same thing will eventually
disagree with it, and when the two disagree, nothing in the repository says
which one is correct. So the question that matters is not what to document.
It is what the code does not, and cannot, state on its own.

## What the code already says

Do not restate any of the following in prose. The code is the only place
these can live without going stale:

- **The signature.** What a function takes and returns, which arguments are
  required, and what the defaults are.
- **The listing.** Which files exist, how they group, and when a directory
  name itself carries meaning.
- **The type.** The shape of a structure and which fields are optional.
- **The test.** Which behavior is guaranteed, and at which boundaries.
- **The manifest.** Which files are centrally managed, which were seeded
  once and now belong to the repo, and what version is pinned.
- **The history.** When something changed, who changed it, and what shipped
  in which release. `git log` and the release notes already carry this.

## What only prose can say

Some things are true about the code but not visible anywhere in it. These
belong in writing, because nothing else will carry them:

- **Why this and not that.** The alternative that was considered and
  rejected, and the grounds for rejecting it. The code shows the decision
  that was made, never the ones that weren't.
- **The invariant.** What has to stay true across future changes, and what
  breaks silently if it stops being true.
- **The boundary.** What is deliberately out of scope, so nobody rebuilds it
  here by accident.
- **Reasoning that spans files.** A property that only holds because several
  modules agree with each other. No single docstring can hold a claim about
  another file.
- **The trap.** The mistake this design prevents. The prevention is
  invisible in the working result; only prose can name what would have gone
  wrong without it.

## When duplication is acceptable

What goes stale is a description of something that changes. A fixed fact
does not change, so repeating it is safe: that an acronym stands for a
particular phrase will not change when the code does, no matter how many
files spell it out.

## The test

One question decides where a piece of documentation belongs: **would this
paragraph need editing if someone renamed a keyword argument?**

If yes, it belongs next to that argument, in a docstring, where it gets
reviewed as part of the same change that moves it. If no, it belongs in
prose, because renaming a keyword argument has no bearing on it.

The test scales. Applied to a sentence, it moves a line into a docstring.
Applied to a table, it deletes an index and points at the code the index
was summarizing. Applied to a page, it closes the file.

## Where it goes

| Content | Destination | How it ages |
|---|---|---|
| Signatures, arguments, defaults | Docstrings | Reviewed with the change that moves it; cannot drift. |
| Why the code is shaped this way, where the reasoning is too broad for one docstring | `docs/*.md` | Ages slowly; correct it when the reasoning changes, not on a schedule. |
| A decision, its context, and its consequences | `docs/adr/NNNN-*.md` | Does not age. Records what was true when it was accepted, not what is true now. |
| What the project is, how to install it, first result | `README.md` | Ages with the public surface; update it when the surface changes. |
| Landmines and conventions an agent needs before editing | `CLAUDE.md` | Instructions, not description. Stable until the convention changes. |
| Concepts shared by more than one repository: the paradigm, the glossary, the ecosystem map | this documentation hub | Linked from every repository, restated in none of them. |
| What a term means | the glossary | One canonical entry per term. |
| Current work, roadmap, known issues, what is planned next | issues and project boards | Live and queryable; closed by the work itself, so it is never stale. |
| Retired plans, superseded diagrams, original prompts | `docs/history/` | Frozen, and read as frozen. Nobody edits it to stay current. |
| License, contributing, security, conduct, support | the shared `actions` templates | Delivered and synced. Never edited locally. |

## What should never be written into a file

Each pattern below was found in a real repository during the pass that this
standard comes from.

- **API restated in prose.** A reference page that lists node kinds,
  arguments, defaults, and versions in a table is a second copy of the
  library's own signatures. Delete the page; improve the docstring instead.
- **A manual file index.** A table listing what's in a directory needs an
  edit on every add or rename, and nothing catches the miss. Let the file
  listing be the index, and name files so the listing reads as one.
- **Plans in the repository.** Known issues, current work, and next
  milestones go stale faster than anything else in a repo, because the work
  itself invalidates them the moment it lands. Separate the plan from the
  intent: what gets built next belongs on a board; what the project is
  building toward, and what it has deliberately decided not to build, stays
  in the repo.
- **An uninformative status line.** "Active development" is true of any
  repository with a recent commit. It tells a reader nothing they couldn't
  get from the commit graph.
- **Duplicate explanation.** The same concept explained in more than one
  place. One copy is always ahead of the other, and nothing marks which.
  Pick one canonical home; everywhere else links to it.
- **Counts and fixed lists.** "The files in this directory" or "the
  supported versions" spelled out as a list goes wrong the moment the set
  changes, and nothing fails when it does. Name the directory or the range
  by rule, not by enumerating its current members.
- **An edited managed file.** A local change to a centrally managed file is
  indistinguishable from staleness: sync can't tell customization from
  drift. Change the template instead.
- **Edit history inside the document.** "This section used to have a table"
  only makes sense to a reader who saw the earlier version. State the
  current rule, not the history of how it got there.

## The other half of the plans rule

Taking the plan out of the repository only works if something replaces it.
That something is:

- **An issue.** What the work is, and when it's done. Anyone can file one,
  but an issue carrying only a title has moved the problem, not solved it.
  The reason, the constraints, and what "done" looks like belong in the
  body.
- **A milestone.** What has to be true for the next release to ship. Name
  it for the outcome, not a date range, so "public alpha" rather than
  "sprint N". Give it a due date, an owner, and close it on that date, with
  anything unfinished moved out by hand. A milestone left to drift past its
  date is a stale list one level up.
- **A project board.** What is in flight now, and in what order.
- **A label.** Only from an agreed set. A label invented for one issue is
  noise, and a board filtered on it silently misses everything else that
  should have carried it.

Most issues will not sit in a milestone, and that's correct. An issue with
no milestone is backlog.

## Acronyms and the glossary

This is the one place where writing the same thing in many files is
correct, because an acronym's expansion is a fixed fact, not a description
that changes.

- **In source files**, expand the acronym on first use in that file, in
  parentheses, including inside comments and docstrings: `TPMS (triply
  periodic minimal surface)`. Lowercase the expansion unless it's a proper
  noun. A reader who lands in the middle of a diff, with no other context,
  still needs to know what the term means.
- **In prose docs**, link the [glossary](glossary.md) instead of
  re-explaining. A reader here can follow a link; a reader in a diff
  cannot.
- **The glossary** holds the canonical entry for any term that needs more
  than an expansion to be understood.

Both halves matter. Expansion without a glossary leaves the concept behind
the acronym undefined. A glossary without expansion leaves every code
reader at a dead end, because someone arriving from a search result or a
diff has nothing to click.

## Warning signs

A document is in the wrong shape or the wrong place if:

- **It needs a checking step.** If keeping a document accurate requires
  someone to manually verify it against the code, encode the property in
  structure instead of prose.
- **A rename would break it.** The text lives too far from the thing it
  describes.
- **Two files share a sentence.** One of them is already stale, and nobody
  has noticed yet.
- **It reads as a proposal.** A decision was made and never recorded.
  Convert it into a decision record with the status it actually has,
  instead of leaving it phrased as an open question.

## What held up

Across a full pass over one repository, the documentation that needed no
correction was the writing that explained *why*. A rationale doesn't change
when a signature does. Everything that went stale was restating something
the code already stated somewhere else.

## Why this standard lives here

This standard lives in the documentation hub as a single page, referenced
from each repository's style guide with one line, rather than copied into
each repository.

The alternative considered was a managed template delivered into every
repository, the way `CONTRIBUTING.md`, `SECURITY.md`, and the rest of the
shared templates are delivered. That has more reach: every repository
would carry a local, always-current copy. But it would place a copy of a
standard about duplication into every repository it governs, which is the
exact pattern this standard prohibits elsewhere. A standard that argues
against restating things cannot be shipped by restating itself.

This is the companion to [`STYLE.md`](../STYLE.md): `STYLE.md` governs how
to write; this page governs where to put it.
