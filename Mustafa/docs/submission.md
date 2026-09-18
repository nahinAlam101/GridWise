# Submission preparation

This project implements the local service. The attached guide also requires externally accessible artifacts; preparing source files does not publish those artifacts.

| Artifact | Current state | Finish before submitting |
| --- | --- | --- |
| HTTP API source | Implemented | Configure a real language model and run live samples |
| Public endpoint | Not deployed | Deploy and verify from outside the development machine |
| Source repository | Local files only | Create the repository under your team account and follow the organizer's timing/visibility rules |
| README/configuration | Included | Record your actual provider/model and deployment details |
| Docker fallback | Dockerfile included; runtime unverified | Build, run, publish, and record an exact pullable tag/digest |
| Architecture video | Narration draft below | Record a real demonstration, keep it under three minutes, and provide an accessible link |

Do not put credential values in a repository, registry image, submitted form, video, or README. Configure them through local environment files or deployment secrets.

## Suggested recording script: approximately 2 minutes 40 seconds

**0:00–0:20 — Problem**

“GridWise schedules one day of campus electricity using the grid, rooftop solar, and a battery. It receives demand, solar forecasts, tariffs, battery limits, and operator notes. Our service first interprets those notes, then finds a low-cost plan that follows every operating constraint.”

Show the request sample and the two required endpoints.

**0:20–0:55 — Language model**

“The operator notes go directly to a language model. It returns one structured directive for every note: solar reduction, minimum battery reserve, charging restriction, discharging restriction, grid cap, or no operation. The prompt distinguishes remaining solar from a percentage reduction and uses start-inclusive, end-exclusive time windows. Irrelevant announcements become no operation.”

Show the configured model name, without showing secrets, and the interpretation response from a real successful call.

**0:55–1:20 — Guardrails**

“We treat model output as untrusted. Deterministic validation checks the note mapping, directive types, exact fields, hours, finite numbers, reserve capacity, and applies flags. Invalid output is repaired within a bounded budget or safely rejected. The model cannot modify demand, tariffs, or battery capacity.”

Show `gridwise/validation.py` and one rejected invalid-output test.

**1:20–1:55 — Optimization and verification**

“A linear program minimizes total grid cost. A signed battery change enforces one action per hour. We apply each directive to the solver's bounds and require the final battery energy to equal its initial energy. Secondary optimization reduces peak intake and battery movement while fixing the optimal cost. An independent replay checks every hour and recomputes all totals before the API returns.”

Show the architecture diagram and a schedule with charging and discharging hours.

**1:55–2:25 — Run and tests**

“The service starts with Python or Docker. The readiness endpoint reports whether the service is configured. Our tests check the public sample costs, all energy constraints, malformed requests, provider failures, and randomized scenarios against an independent optimal-cost oracle.”

Run `/health`, one live optimization, and `python scripts/check_samples.py --url YOUR_URL`. Only claim live results that actually pass while recording; offline tests use supplied reference interpretations.

**2:25–2:40 — Reproducibility**

“The README contains the exact environment variables, run commands, tests, model configuration, and Docker fallback instructions. Credentials are provided at runtime. The submitted endpoint and pullable image let organizers repeat the same checks.”

Show the actual endpoint and image reference after deployment. End before 3:00.
