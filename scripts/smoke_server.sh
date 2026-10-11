#!/usr/bin/env bash
set -euo pipefail

PORT="${SMOKE_PORT:-8901}"
BASE="http://127.0.0.1:${PORT}"
TOKEN="smoke-token"

BIG=$(mktemp)
PAGE=$(mktemp)
NOVA_BACKEND=null NOVA_API_TOKEN="$TOKEN" NOVA_RATE_LIMIT=100 PORT="$PORT" python server.py &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; rm -f "$BIG" "$PAGE"' EXIT

curl -sf --retry 20 --retry-connrefused --retry-delay 1 -o /dev/null "$BASE/" \
  || { echo "FAIL server did not boot"; exit 1; }
kill -0 "$SERVER_PID" 2>/dev/null || { echo "FAIL server process died; something else owns port ${PORT}"; exit 1; }

expect() {
  local want=$1 label=$2 got; shift 2
  got=$(curl -s -o /dev/null -w "%{http_code}" "$@") || got=000
  [ "$got" = "$want" ] || { echo "FAIL ${label}: want ${want}, got ${got}"; exit 1; }
  echo "ok   ${label} -> ${got}"
}

JSON='Content-Type: application/json'
expect 200 "index served"    "$BASE/"
expect 404 "no source leak"  "$BASE/server.py"
expect 401 "token required"  -X POST "$BASE/api/generate" -H "$JSON" -d '{"text":"hello there"}'
expect 400 "unknown style"   -X POST "$BASE/api/generate" -H "$JSON" -H "X-API-Token: $TOKEN" \
  -d '{"text":"hello there","style":"<script>"}'
expect 400 "Infinity seed"   -X POST "$BASE/api/generate" -H "$JSON" -H "X-API-Token: $TOKEN" \
  -d '{"text":"hello there","seed": Infinity}'
expect 400 "oversized seed"  -X POST "$BASE/api/generate" -H "$JSON" -H "X-API-Token: $TOKEN" \
  -d '{"text":"hello there","seed": 9223372036854775808}'
expect 400 "non-dict body"   -X POST "$BASE/api/generate" -H "$JSON" -H "X-API-Token: $TOKEN" -d '[1,2]'

python -c "print('{\"text\": \"' + 'x' * 33800 + '\"}')" > "$BIG"
expect 413 "oversize body"   -X POST "$BASE/api/analyze" -H "$JSON" --data-binary @"$BIG"

curl -s "$BASE/" > "$PAGE"
grep -q tokenInput "$PAGE" || { echo "FAIL frontend token field missing"; exit 1; }
echo "ok   frontend token field present"
echo "SMOKE PASS"
