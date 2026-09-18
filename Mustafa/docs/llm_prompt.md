# Operator-note interpretation

`gridwise/interpreter.py` contains the exact production system prompt and JSON
Schema. Every uncached request calls a configured generative language model via
`POST {LLM_BASE_URL}/chat/completions`. The model directly produces all six
directive types. There is no regular-expression interpreter or public-case
lookup in the production request path.

The prompt specifies one ordered result per note, exact adjustment shapes,
end-exclusive whole-hour windows, noon/midnight conversion, usable solar fraction
remaining, reserve percentages converted with the actual battery capacity, and
irrelevant or explicitly other-day notes. Notes are enclosed as JSON data and
treated as untrusted; embedded instructions cannot change the system contract.
These instructions improve extraction but do not prove semantic correctness.

The default `gpt-4.1-mini` supports Chat Completions and Structured Outputs,
according to its [official model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
The default `LLM_JSON_MODE=json_schema` uses `strict: true` and a nested union of
the six exact response shapes, following the
[Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs).
For another OpenAI-compatible provider that only supports JSON mode, explicitly
set `LLM_JSON_MODE=json_object`. Its schema is also included in the prompt, and
the same deterministic validation remains mandatory.

Model output is rejected for malformed or duplicate JSON keys, truncation,
refusal, wrong wrapper keys, invalid mappings, unsupported types, bad hours,
wrong adjustment shapes, and invalid numeric values. A failed extraction receives
one correction request by default. This never silently removes a constraint or
replaces the LLM with a local guess. Provider failures return controlled errors
without exposing credentials, response bodies, or stack traces.

Defaults are a 10-second provider timeout, two total attempts, 1,800 maximum
completion tokens, and a hard 22-second interpretation deadline, leaving time
for deterministic optimization and replay within the competition's 30 seconds.
Set `LLM_TIMEOUT_SECONDS`, `LLM_MAX_ATTEMPTS` (1–3), and
`LLM_MAX_OUTPUT_TOKENS` (256–4096) to tune the bounded request. The hard deadline
still applies. LLM throughput and account limits can affect latency and success.

Only validated responses enter a 256-entry in-memory LRU cache. Its key hashes
the exact notes and every battery field supplied to the model. Scenario IDs are
not used, and changing battery capacity forces fresh interpretation. Results are
copied on return so callers cannot corrupt cached constraints.

The interpreter reads the repository-root `.env` without overriding existing
environment variables. Set `LLM_API_KEY` (or `OPENAI_API_KEY`), optionally
`LLM_BASE_URL` and `LLM_MODEL`. Provider credentials stay in the HTTP Authorization
header and never appear in prompts. A local compatible service can be used by
configuring its base URL and model; a missing key is allowed only for localhost,
127.0.0.1, and ::1. Remote providers require a key. Readiness reports only
configuration presence; it cannot certify provider account access without a
real request. Local tests use mocked HTTP responses and do not incur LLM charges;
a live model smoke test is still needed with the team's chosen provider.
