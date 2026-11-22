import { Board, parseFen } from "./board.js";

// ------------------------------------------------------------------ config

const LEVELS = [
  {
    id: "instinct",
    label: "Instinct",
    instinct: true,
    think: 0.1, // ignored by the server in instinct mode
    desc: "No search. The network plays its first instinct from a single forward pass.",
  },
  { id: "fast", label: "Fast", think: 0.3, desc: "About 0.3 seconds of search per move." },
  { id: "normal", label: "Normal", think: 1.0, desc: "About one second of search per move." },
  { id: "deep", label: "Deep", think: 3.0, desc: "Up to three seconds per move. The strongest setting." },
];

const PIECE_VALUES = { P: 1, N: 3, B: 3, R: 5, Q: 9, K: 0 };
const START_COUNTS = { P: 8, N: 2, B: 2, R: 2, Q: 1 };
const ARROW_COLORS = ["#d9622b", "#5b8bd9", "#6fae7b"];
const MIN_ENGINE_DELAY_MS = 350;

const state = {
  moves: [],
  position: null,
  human: "w",
  colorChoice: "w",
  level: LEVELS[2],
  busy: false,
  watching: false,
  token: 0, // bumped on new game / undo so stale responses are ignored
};

const $ = (sel) => document.querySelector(sel);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ api

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      msg = (await res.json()).detail || msg;
    } catch (_) {
      /* not json */
    }
    throw new Error(msg);
  }
  return res.json();
}

// ------------------------------------------------------------------ board

const board = new Board($("#board"), {
  onMove: (uci) => onHumanMove(uci),
  onPromotion: (color) => pickPromotion(color),
});

function pickPromotion(color) {
  const modal = $("#promo");
  const card = modal.querySelector(".promo-card");
  card.textContent = "";
  return new Promise((resolve) => {
    const done = (piece) => {
      modal.hidden = true;
      modal.onclick = null;
      resolve(piece);
    };
    for (const p of ["q", "r", "b", "n"]) {
      const btn = document.createElement("button");
      btn.style.backgroundImage = `url(/static/pieces/${color}${p.toUpperCase()}.svg)`;
      btn.setAttribute("aria-label", { q: "Queen", r: "Rook", b: "Bishop", n: "Knight" }[p]);
      btn.onclick = (e) => {
        e.stopPropagation();
        done(p);
      };
      card.append(btn);
    }
    modal.onclick = () => done(null);
    modal.hidden = false;
  });
}

// ------------------------------------------------------------------ game flow

const turnAfter = (plies) => (plies % 2 === 0 ? "w" : "b");

function lastMoveSquares() {
  const uci = state.moves[state.moves.length - 1];
  return uci ? { from: uci.slice(0, 2), to: uci.slice(2, 4) } : null;
}

function applyPosition(pos) {
  state.position = pos;
  board.setPosition(pos.fen, { lastMove: lastMoveSquares(), check: pos.check });
  const humanToMove = !state.watching && !state.busy && pos.turn === state.human && !pos.status.over;
  board.setLegalMoves(humanToMove ? pos.legal : []);
  board.setInteractive(humanToMove);
  renderMoveList(pos.history);
  renderPlayers(pos.fen);
  renderStatus();
  if (pos.status.over) showGameOver(pos.status);
}

async function refreshPosition() {
  applyPosition(await api("/api/position", { moves: state.moves }));
}

async function onHumanMove(uci) {
  if (state.busy) return;
  const token = state.token;
  board.setArrows([]);
  state.moves.push(uci);
  try {
    const pos = await api("/api/position", { moves: state.moves });
    if (token !== state.token) return;
    applyPosition(pos);
    if (!pos.status.over) await engineMove();
  } catch (err) {
    state.moves.pop();
    toast(err.message);
    refreshPosition();
  }
}

async function engineMove() {
  const token = state.token;
  const level = state.level;
  setBusy(true);
  const started = performance.now();
  try {
    const res = await api("/api/move", {
      moves: state.moves,
      think_time: level.think,
      instinct: !!level.instinct,
    });
    const wait = MIN_ENGINE_DELAY_MS - (performance.now() - started);
    if (wait > 0) await sleep(wait);
    if (token !== state.token) return;

    state.moves.push(res.move);
    renderAnalysis(res.analysis, { mode: level.instinct ? "instinct" : "engine" });
    setBusy(false);
    applyPosition(res.position);
  } catch (err) {
    if (token !== state.token) return;
    setBusy(false);
    toast(`Engine error: ${err.message}`);
    renderStatus();
  }
}

async function newGame() {
  state.token++;
  state.watching = false;
  state.busy = false;
  state.moves = [];
  state.human = state.colorChoice === "r" ? (Math.random() < 0.5 ? "w" : "b") : state.colorChoice;
  hideGameOver();
  board.setArrows([]);
  board.setOrientation(state.human);
  resetAnalysis();
  updateWatchButton();
  await refreshPosition();
  if (state.human === "b") await engineMove();
}

async function undo() {
  if (state.watching || !state.moves.length) return;
  state.token++;
  state.busy = false;
  do {
    state.moves.pop();
  } while (state.moves.length && turnAfter(state.moves.length) !== state.human);

  hideGameOver();
  board.setArrows([]);
  await refreshPosition();
  if (turnAfter(state.moves.length) !== state.human) await engineMove();
}

async function hint() {
  if (state.busy || state.watching || !state.position || state.position.status.over) return;
  if (state.position.turn !== state.human) return;
  const token = state.token;
  setBusy(true, "Looking for a good move for you...");
  try {
    const res = await api("/api/analyse", { moves: state.moves, think_time: Math.max(1, state.level.think) });
    if (token !== state.token || !res.analysis) return;
    renderAnalysis(res.analysis, { mode: "hint" });
    board.setArrows(arrowsFor(res.analysis));
  } catch (err) {
    toast(err.message);
  } finally {
    if (token === state.token) {
      setBusy(false);
      applyPosition(state.position);
    }
  }
}

async function toggleWatch() {
  if (state.watching) {
    state.watching = false;
    updateWatchButton();
    return;
  }
  if (state.position?.status.over) await newGame();
  state.watching = true;
  hideGameOver();
  board.setArrows([]);
  updateWatchButton();
  applyPosition(state.position);
  while (state.watching && !state.position.status.over) {
    await engineMove();
    await sleep(250);
  }
  state.watching = false;
  updateWatchButton();
  applyPosition(state.position);
}

function setBusy(busy, message) {
  state.busy = busy;
  $("#think-pill").hidden = !busy;
  if (busy) {
    board.setInteractive(false);
    $("#status").textContent = message || "Engine is thinking...";
  }
}

// ------------------------------------------------------------------ rendering

function formatScore(score) {
  if (score.mate !== null && score.mate !== undefined) {
    return score.mate > 0 ? `#${score.mate}` : `-#${Math.abs(score.mate)}`;
  }
  const v = score.cp / 100;
  return `${v > 0 ? "+" : ""}${v.toFixed(2)}`;
}

function describeScore(score) {
  if (score.mate !== null && score.mate !== undefined) {
    const side = score.mate > 0 ? "White" : "Black";
    return `${side} has mate in ${Math.abs(score.mate)}.`;
  }
  const cp = score.cp;
  const side = cp > 0 ? "White" : "Black";
  const a = Math.abs(cp);
  if (a < 30) return "Roughly equal.";
  if (a < 100) return `A slight edge for ${side}.`;
  if (a < 300) return `${side} is better.`;
  return `${side} is winning.`;
}

function renderEvalBar(score) {
  const bar = $("#evalbar");
  const pct = Math.round(score.win * 1000) / 10;
  bar.querySelector(".evalbar-fill").style.height = `${pct}%`;
  bar.setAttribute("aria-valuenow", String(pct));
  bar.classList.toggle("black-ahead", score.win < 0.5);
  bar.classList.toggle("flipped", board.orientation === "b");
  let label;
  if (score.mate !== null && score.mate !== undefined) label = `M${Math.abs(score.mate)}`;
  else {
    const pawns = Math.abs(score.cp / 100);
    label = pawns >= 10 ? pawns.toFixed(0) : pawns.toFixed(1);
  }
  $("#evalbar-label").textContent = label;
}

function resetAnalysis() {
  $("#eval-big").textContent = "0.00";
  $("#eval-words").textContent = "Waiting for the first move.";
  for (const id of ["st-depth", "st-nodes", "st-evals", "st-time"]) $(`#${id}`).textContent = "-";
  $("#lines").innerHTML = '<li class="empty">No search yet.</li>';
  $("#policy").innerHTML = '<li class="empty">-</li>';
  $("#explain").textContent = "";
  renderEvalBar({ cp: 0, mate: null, win: 0.5 });
}

function renderAnalysis(a, { mode }) {
  $("#eval-big").textContent = formatScore(a.score);
  $("#eval-words").textContent = describeScore(a.score);
  renderEvalBar(a.score);

  $("#st-depth").textContent = a.depth === 0 ? "0" : String(a.depth);
  $("#st-nodes").textContent = a.nodes.toLocaleString();
  $("#st-evals").textContent = a.nn_evals.toLocaleString();
  $("#st-time").textContent = `${(a.time_ms / 1000).toFixed(2)}s`;

  const lines = $("#lines");
  lines.textContent = "";
  a.lines.slice(0, 3).forEach((line, i) => {
    const li = document.createElement("li");
    if (i === 0) li.className = "best";
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = line.exact ? formatScore(line) : `<= ${formatScore(line)}`;
    const pv = document.createElement("span");
    pv.className = "pv";
    const parts = (line.pv || line.san).split(" ");
    // Bold the candidate move itself (first token that is not a move number).
    let bolded = false;
    parts.forEach((tok, j) => {
      if (j) pv.append(" ");
      if (!bolded && !/^\d+\.+$/.test(tok)) {
        const b = document.createElement("b");
        b.textContent = tok;
        pv.append(b);
        bolded = true;
      } else {
        pv.append(tok);
      }
    });
    const prior = document.createElement("span");
    prior.className = "prior";
    prior.textContent = `${(line.prior * 100).toFixed(0)}%`;
    prior.title = "Network policy probability for this move";
    li.append(score, pv, prior);
    lines.append(li);
  });

  const policy = $("#policy");
  policy.textContent = "";
  const top = a.policy.slice(0, 6);
  const max = Math.max(...top.map((p) => p.prob), 0.001);
  for (const p of top) {
    const li = document.createElement("li");
    if (p.move === a.best_move) li.className = "chosen";
    li.innerHTML = `<span class="san"></span><span class="bar"><span></span></span><span class="pct"></span>`;
    li.querySelector(".san").textContent = p.san;
    li.querySelector(".bar span").style.width = `${(p.prob / max) * 100}%`;
    li.querySelector(".pct").textContent = `${(p.prob * 100).toFixed(1)}%`;
    policy.append(li);
  }

  $("#explain").textContent = explain(a, mode);
}

function explain(a, mode) {
  const staticEval = formatScore({ cp: a.static_eval, mate: null });
  const before = ` Before searching, the network rated the position ${staticEval} and gave White a ${Math.round(
    a.result_head_win * 100,
  )}% expected score.`;

  if (mode === "instinct") {
    return `Instinct mode: no search was run, so ${a.best_san} is simply the network's top choice.${before}`;
  }

  const subject = mode === "hint" ? `The search suggests ${a.best_san}` : `The engine played ${a.best_san}`;
  const rank = a.policy.findIndex((p) => p.move === a.best_move);
  const prior = ((a.lines[0]?.prior || 0) * 100).toFixed(1);
  const depth = `${a.depth} ply`;

  if (rank === 0) {
    return `${subject}, which was also the network's first instinct; searching ${depth} ahead confirmed it.${before}`;
  }
  const which = rank > 0 ? `the network's #${rank + 1} choice` : "a move outside the network's top picks";
  return `${subject}, ${which} at ${prior}%. Searching ${depth} ahead overruled the network's favourite, ${a.policy[0].san}.${before}`;
}

function arrowsFor(a) {
  return a.lines.slice(0, 3).map((line, i) => ({
    from: line.move.slice(0, 2),
    to: line.move.slice(2, 4),
    color: ARROW_COLORS[i],
    width: i === 0 ? 2.6 : 1.8,
    opacity: i === 0 ? 0.9 : 0.65,
  }));
}

function renderMoveList(history) {
  const ol = $("#movelist");
  ol.textContent = "";
  if (!history.length) {
    const li = document.createElement("li");
    li.innerHTML = '<span class="placeholder">No moves yet.</span>';
    ol.append(li);
    return;
  }
  for (let i = 0; i < history.length; i += 2) {
    const li = document.createElement("li");
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = `${i / 2 + 1}.`;
    li.append(num);
    for (const j of [i, i + 1]) {
      const mv = document.createElement("span");
      mv.className = "mv";
      if (j < history.length) mv.textContent = history[j];
      if (j === history.length - 1) mv.classList.add("current");
      li.append(mv);
    }
    ol.append(li);
  }
  ol.scrollTop = ol.scrollHeight;
}

function renderPlayers(fen) {
  const pieces = parseFen(fen);
  const counts = { w: {}, b: {} };
  let material = 0; // positive = white ahead
  for (const code of pieces.values()) {
    counts[code[0]][code[1]] = (counts[code[0]][code[1]] || 0) + 1;
    material += (code[0] === "w" ? 1 : -1) * PIECE_VALUES[code[1]];
  }

  const bottom = board.orientation;
  const top = bottom === "w" ? "b" : "w";
  const name = (color) => (state.watching || color !== state.human ? "Engine" : "You");

  for (const [stripId, color] of [["#strip-top", top], ["#strip-bottom", bottom]]) {
    const strip = $(stripId);
    strip.querySelector(".player-dot").dataset.color = color;
    strip.querySelector(".player-name").textContent = name(color);
    const sub = strip.querySelector(".player-sub");
    if (sub) sub.textContent = name(color) === "Engine" ? state.level.label : "";

    // Pieces this side has captured = opponent pieces missing from the board.
    const opp = color === "w" ? "b" : "w";
    const box = strip.querySelector(".captured");
    box.textContent = "";
    for (const type of ["Q", "R", "B", "N", "P"]) {
      const missing = Math.max(0, START_COUNTS[type] - (counts[opp][type] || 0));
      for (let k = 0; k < missing; k++) {
        const img = document.createElement("img");
        img.src = `/static/pieces/${opp}${type}.svg`;
        img.alt = "";
        box.append(img);
      }
    }
    const adv = color === "w" ? material : -material;
    if (adv > 0) {
      const span = document.createElement("span");
      span.className = "adv";
      span.textContent = `+${adv}`;
      box.append(span);
    }
  }
}

function renderStatus() {
  const pos = state.position;
  if (!pos) return;
  const el = $("#status");
  if (pos.status.over) {
    el.textContent = resultText(pos.status).title;
  } else if (state.busy) {
    el.textContent = "Engine is thinking...";
  } else if (state.watching) {
    el.textContent = "Watching the engine play itself.";
  } else if (pos.turn === state.human) {
    el.textContent = pos.check ? "You are in check. Your move." : "Your move.";
  } else {
    el.textContent = "Engine to move.";
  }
}

function resultText(status) {
  const reason = status.reason;
  if (status.result === "1/2-1/2") return { title: "Draw", sub: `By ${reason}.` };
  const winner = status.result === "1-0" ? "w" : "b";
  const winnerName = winner === "w" ? "White" : "Black";
  if (state.watching || !state.moves.length) return { title: `${winnerName} wins`, sub: `By ${reason}.` };
  return winner === state.human
    ? { title: "You win", sub: `${winnerName} wins by ${reason}.` }
    : { title: "The engine wins", sub: `${winnerName} wins by ${reason}.` };
}

function showGameOver(status) {
  const { title, sub } = resultText(status);
  $("#game-over-title").textContent = title;
  $("#game-over-sub").textContent = sub;
  $("#game-over").hidden = false;
}

function hideGameOver() {
  $("#game-over").hidden = true;
}

function updateWatchButton() {
  const btn = $("#watch-btn");
  btn.textContent = state.watching ? "Stop" : "Watch";
  btn.classList.toggle("active", state.watching);
}

let toastTimer = null;
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 2600);
}

async function copy(text, label) {
  try {
    await navigator.clipboard.writeText(text);
    toast(`${label} copied to clipboard.`);
  } catch (_) {
    window.prompt(`Copy the ${label}:`, text);
  }
}

// ------------------------------------------------------------------ controls

function setupSegmented(container, items, isActive, onPick) {
  container.textContent = "";
  for (const item of items) {
    const btn = document.createElement("button");
    btn.setAttribute("role", "radio");
    btn.textContent = item.label;
    btn.onclick = () => {
      onPick(item);
      for (const b of container.children) b.setAttribute("aria-checked", "false");
      btn.setAttribute("aria-checked", "true");
    };
    btn.setAttribute("aria-checked", String(isActive(item)));
    container.append(btn);
  }
}

setupSegmented(
  $("#level-picker"),
  LEVELS,
  (l) => l === state.level,
  (l) => {
    state.level = l;
    $("#level-desc").textContent = l.desc;
    if (state.position) renderPlayers(state.position.fen);
  },
);
$("#level-desc").textContent = state.level.desc;

for (const btn of $("#color-picker").children) {
  btn.onclick = () => {
    state.colorChoice = btn.dataset.color;
    for (const b of $("#color-picker").children) b.setAttribute("aria-checked", String(b === btn));
  };
}

document.addEventListener("click", (e) => {
  const action = e.target.closest("[data-action]")?.dataset.action;
  if (!action) return;
  switch (action) {
    case "new":
    case "rematch":
      newGame();
      break;
    case "undo":
      undo();
      break;
    case "hint":
      hint();
      break;
    case "flip":
      board.flip();
      if (state.position) renderPlayers(state.position.fen);
      $("#evalbar").classList.toggle("flipped", board.orientation === "b");
      break;
    case "watch":
      toggleWatch();
      break;
    case "dismiss":
      hideGameOver();
      break;
    case "copy-pgn":
      if (state.position) copy(state.position.pgn, "PGN");
      break;
    case "copy-fen":
      if (state.position) copy(state.position.fen, "FEN");
      break;
  }
});

// ------------------------------------------------------------------ start

async function loadInfo() {
  try {
    const info = await api("/api/info");
    const n = info.network;
    const params = (n.parameters / 1e6).toFixed(1);
    const arch = n.arch === "cbam" ? "residual blocks with CBAM attention" : "residual blocks";
    $("#net-meta").textContent =
      `Live: ${params}M parameters, ${n.num_blocks} ${arch} at ${n.trunk_channels} channels (${n.checkpoint}), running on ${n.device}.`;
    const s = info.search;
    $("#search-meta").textContent =
      `Live: MultiPV ${s.multipv}, ${s.prune_width} moves per node plus tactics, batches of ${s.batch_size}.`;
  } catch (_) {
    /* the about section still reads fine without live numbers */
  }
}

resetAnalysis();
loadInfo();
newGame().catch((err) => {
  $("#status").textContent = `Could not reach the engine: ${err.message}`;
});
