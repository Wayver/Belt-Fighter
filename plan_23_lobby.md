# plan_23 — Lobby: find a match, then play P2P

## Goal
Let two players connect to a shared **lobby server** and play against each
**P2P** from there, without one player having to know the other's LAN IP.

## Non-goals (v1)
- The lobby is NOT a game server. It holds no game state, relays no game
  traffic, and is not on the game's critical path. Once two players are
  matched, they play the existing host-authoritative 2P game **directly**
  (the host's `NET_PORT` socket, the `join`/`welcome` handshake, snapshots,
  input, echoes — all unchanged).
- No accounts, no chat, no matchmaking, no NAT traversal. If the host is
  behind a NAT that hides its LAN IP, the direct connect fails — same
  limitation as today's typed-IP join. (A VPS lobby + a public host IP is
  the practical deployment; hole-punching is a future plan.)
- No reconnect / no lobby persistence (pinned #8 applies to the game; the
  lobby is stateless-per-connection and forgets rooms when a peer drops).

## Topology
```
                 lobby server (standalone process, any machine)
                 listens on LOBBY_PORT (7778)
                    ^                ^
     lobby conn     |                |     lobby conn
                    |                |
        [host app]  |                |  [client app]
        run_host    |                |  run_client
             \      |                |      /
              \     |                |     /
               \    |                |    /
                v   v                v   v
                 +------------------------+
                 |  DIRECT P2P game      |
                 |  host NET_PORT socket |
                 |  (existing netcode)   |
                 +------------------------+
```
- **Host app**: spawns its lobby process, sends `create` (carries its
  public `ip:port` — the same `_lan_ip()` + `NET_PORT` the waiting screen
  shows today), receives `match` with the client's `ip:port`, then runs
  `run_host` exactly as it does today (binds `NET_PORT`, waits for the
  client's direct connect).
- **Client app**: spawns its lobby process, sends `join` (no payload),
  receives `match` with the host's `ip:port`, then runs `run_client`
  against that address — exactly today's join path.
- The lobby's only job: pair one `create` with one `join` and hand each
  side the other's address.

## Why a process (client side)
Same rule as `net_process.py` / `client_net_process.py`: socket I/O runs in
a spawned child with its own GIL, off the render thread. The render process
talks to it only through a status Pipe (non-blocking `poll()`), and the
child exits on `("quit",)` or Pipe EOF. The lobby connection is short-lived
(a few seconds at most), so the child is spawned, does its job, and is
shut down — it never coexists with the game's net process.

## Wire protocol (lobby)
Same framing as the game wire: 4-byte big-endian length + JSON, via
`net.encode_frame` / `net.extract_frames` / `net.Connection`. Reusing
`Connection` means the partial-frame send + receive-resync behavior comes
for free.

Messages (the `type` field):
```
create  (client->lobby)  {"type":"create","addr":"ip:port"}
join    (client->lobby)  {"type":"join"}
match   (lobby->client)  {"type":"match","addr":"ip:port"}   # the peer's addr
full    (lobby->client)  {"type":"full"}                     # no open room to join
closed  (lobby->client)  {"type":"closed"}                   # your room was cancelled
error   (lobby->client)  {"type":"error","msg":"..."}        # malformed payload
```
- One open room at a time on the server (2P game, one host at a time).
  `create` while a room is open -> `error` ("room already open").
  `join` with no open room -> `full`.
- On `match`, the server closes the room. A later `join` gets `full`.
- If a matched peer's connection drops before its `match` is flushed, the
  server sends `closed` to the other side and clears the room. (Best effort
  — the game's own handshake timeout covers the rest.)
- `addr` is a plain `"ip:port"` string. The server does NOT validate it
  beyond being a non-empty string (the game's connect phase fails cleanly
  on a bad address, and the lobby trusts the host's self-reported address
  — same trust model as the typed-IP join today).

## Modules
- `lobby.py`
  - `LobbyServer(port)` — a small thread-per-connection server (stdlib
    `socket` + `threading`). One worker thread per connected client, each
    owning its `Connection`. Shared state (the single open room) is guarded
    by a `threading.Lock`. Headless: no pygame import (only `net` framing
    is imported, and only its pure functions — see the import note below).
  - `lobby_client_main(status_conn, lobby_ip, lobby_port, action, addr)` —
    the SPAWN TARGET (top-level function, like `net_process_main`).
    Connects to the lobby, sends `create`/`join`, waits for `match`/
    `full`/`closed`/`error`, reports on the status Pipe, exits.
    Status messages to the render process:
      `("listening",)` — not used by the client (host waiting screen only)
      `("match", "ip:port")` — matched; the peer's direct address
      `("full",)` — no open room (join)
      `("closed",)` — the room was cancelled
      `("error", text)` — connect failed / lobby said error
      `("crash", traceback_text)`
    Exits on `("quit",)` or status-Pipe EOF.
  - `main()` — `python -m ship5.lobby [port]` runs the server standalone.
- `test_lobby.py` — headless test (run `python -m ship5.test_lobby`):
  spawns a real `LobbyServer` on a free port in-process, spawns TWO real
  `lobby_client_main` child processes (spawn context, like the other net
  tests), and verifies the full create/join/match relay + the error paths.

### Import note (headless)
`net.py` imports `intent` -> `pygame`, so anything that imports `net` must
set `SDL_VIDEODRIVER=dummy` first (the established pattern). The server
itself only needs `encode_frame`/`extract_frames`/`Connection` — all pure
socket/JSON — but they live in `net.py`, so the server process sets the
dummy drivers too (harmless; it never creates a display). The client spawn
target sets them inside `lobby_client_main` (so the parent keeps its real
video driver), exactly like `net_process_main`.

## Milestones
- **M1 (DONE)**: `lobby.py` (server + client spawn target) +
  `test_lobby.py` proving the relay end-to-end over real loopback TCP with
  real spawned processes. No game/menu changes. Verified: `test_lobby.py`
  ALL PASS (match relay, full/room-already-open/post-match-full/malformed/
  unknown-type error paths, server smoke), `net` self-test +
  `test_net_process.py` still ALL PASS (no regression), standalone
  `python -m ship5.lobby [port]` binds + listens.
- **M2 (next session)**: menu integration — a LOBBY mode on the mode
  screen (sub-choice: host-via-lobby / join-via-lobby, then the existing
  hull/loadout flow), `run_host`/`run_client` accept an optional
  pre-resolved address, and the lobby process is spawned + polled around
  the existing waiting/connecting screens. Direct IP host/join stays as-is.

## Risks / open questions
- **NAT**: the lobby hands out the host's self-reported address. Behind a
  NAT this is a private IP the client can't reach. v1 accepts this (same
  as typed-IP join); a VPS-hosted lobby + a host with a reachable address
  is the intended deployment.
- **Server trust**: the lobby relays whatever address a host claims. Fine
  for friends-on-a-LAN / a VPS you run; not a security boundary.
- **One room at a time** keeps the server trivially correct. Multiple
  concurrent rooms is a small future extension (dict of rooms).