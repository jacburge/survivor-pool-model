# Hosting Plan: Multi-User Survivor Pool Tool

This is the working plan for Phase 9 of [plan.md](plan.md) — turning the
single-user CLI/scripts pipeline into a small hosted web app a small group
of people can each use for their own league. It's a companion to Phase 9,
not a replacement: Phase 9 there stays the one-paragraph summary tracked
alongside the other phases; this file is where we work out the actual
decisions.

Add questions/comments inline marked with `//` — that's the plan.

## Context

- **Scale:** a small group of people, across a handful of leagues. Not a
  public signup product. A league can have more than one person in it —
  e.g. two people who are both entrants in the same real-world Splash pool
  — sharing that pool's field simulation while each person gets their own
  entries, their own entry cap, and their own recommended allocation
  scored independently.
- **Timeline:** soon — targeting live for Week 4 or shortly after, in
  parallel with, not blocking, Phase 7's October 4 lock-day submission.
- **Starting point:** Railway (see below) — Megan already has an account
  and hosts SQL there.
- **Out of scope for this document:** the Splash picksheet auto-scraper
  (separate thread, needs the actual picksheet URL before it can be
  designed) — rival tracking stays manual either way, hosting doesn't
  change that.
- **Team:** other people will be working on this project too, not just
  Megan — worth keeping in mind for anything access-related (repo,
  Railway project, credentials).

## Platform: Railway, now a real ~$5/mo rather than free

Railway hosting requires a $5/mo membership once free resources are utilized. Megan has used this for PWHL analysis and has a $5 account that can be used, although as PWHL season begins (Nov) usage may heighten to need a $10/mo account.

- **Deploy:** git push deploy (Nixpacks auto-detects the Python app), no
  Dockerfile required.
- **Postgres:** managed add-on — can live in the same project as Megan's
  existing SQL hosting.
- **Scheduled jobs:** set a cron schedule directly on a service, instead of
  running it always-on.
- **Cost at this scale:** the Hobby plan (~$5/mo) includes a usage credit
  that should easily cover a low-traffic app like this. Worth
  double-checking current pricing in Megan's dashboard since it's
  usage-based and plans change.

**Recommendation: still Railway, at ~$5/mo — confirm that's fine before
committing to it.** Worth keeping an eye on combined usage once PWHL season
picks up in November, since that's the same account.

## Frontend: server-rendered pages, or a SPA a designer builds

Open item, not yet decided — a coworker who writes code may want to build
the UI, possibly as a separate single-page app (SPA) rather than the
server-rendered pages the MVP scope below assumes. Framework not chosen
yet.

This barely changes the plan, because the backend was always going to be a
real API, not a pile of logic tangled into HTML templates:

- **Doesn't change:** build order steps 1-3 (data model, wrapping the
  `survivor/` pipeline, the weekly job script) are backend-only and don't
  care what the frontend is. That work can start now regardless of her
  framework choice.
- **Changes slightly:** step 2's FastAPI app returns JSON instead of
  rendering Jinja2 HTML — simpler, not more work, since it skips writing
  throwaway templates that would just get replaced.
- **Splits into two tracks:** step 4 (currently "auth + the one dashboard
  page") becomes backend auth endpoints (invite-code signup, issuing a
  session/token) plus a separate frontend app calling that API — buildable
  in parallel once the API exists.
- **On Railway:** a project can hold multiple services, so the FastAPI
  backend and the SPA's static build become two services in one project,
  not two separate hosting setups.
- **One real decision to make, not urgent:** session cookies (fits
  server-rendered pages naturally) vs token-based auth (what a SPA usually
  wants) — pick once a framework is chosen, and be consistent about it.

//

## MVP scope

Cutting this down hard rather than building the full Phase 9 vision at
once.

### In v1

- League config (entry cap, buy-in, rake, pot, week range) — replaces the
  hardcoded constants in `survivor/decision/portfolio.py`
- Accounts via invite-code signup (Megan creates a league, generates a
  code, a person signs up with it) — no OAuth, no password-reset flow yet
- Multiple people can belong to the same league; each person's entries and
  recommendation are their own, scored against that league's shared field
  simulation
- One page per league: this week's recommended allocation, reusing
  `best_allocations` directly
- A scheduled job that runs the existing pipeline weekly and stores the
  result

### Out of v1 (later, not blocking)

- The Splash auto-scraper — rival tracking stays manual, same as today
- Any UI for pick history, split-decision logic, the endgame solver —
  Phase 8 items, unaffected by hosting
- Self-serve league creation UI — Megan creating leagues by hand for a few
  people is fine at this scale

## Build order

1. **Data model.** New Postgres tables: `users`, `leagues`,
   `league_memberships` (many-to-many — a league can have several people in
   it, e.g. two people sharing the same real-world pool), `entries` (scoped
   to a user *within* a league, with that league's entry cap enforced per
   person), `recommendations`. Ratings/odds/schedule caches stay as-is for
   now (global, not per-league) — moving those into the DB too isn't
   necessary for v1.
2. **Wrap the pipeline, don't rewrite it.** A FastAPI app that calls
   `survivor/`'s existing functions directly — `best_allocations`,
   `simulate_rival_field`, etc. — and serves JSON. Nothing in `survivor/`
   needs to change for this, and this holds regardless of what the
   frontend ends up being (see "Frontend" above).
3. **Weekly job script.** Refresh global data once, then loop over active
   leagues, size the field sim to each league's rival count, score each
   member's entries, store to `recommendations`.
4. **Auth endpoints + UI.** Invite-code signup, session/token issuing on
   the backend; a page showing a league's current recommendation on the
   frontend — server-rendered or a separate SPA, per the open decision
   above. These two halves can be built in parallel once step 2's API
   exists.
5. **Deploy to Railway**, wire a scheduled service (Railway's cron
   schedule) to the weekly script at the two Phase 8 trigger points
   (post-Thursday-kickoff, post-Sunday-lock).
6. **Invite people.**

Steps 1-3 are the real work; 4-6 are comparatively quick once those exist.

