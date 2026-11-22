// A small, dependency-free chess board.
//
// The board knows nothing about chess rules. It is told the position (FEN),
// which moves are legal, and calls back with a UCI string when the user makes
// one of them. Rules live on the server (python-chess), so there is exactly
// one source of truth.

const FILES = "abcdefgh";
const SVG_NS = "http://www.w3.org/2000/svg";

export function parseFen(fen) {
  const placement = fen.split(" ")[0];
  const map = new Map();
  placement.split("/").forEach((row, i) => {
    const rank = 8 - i;
    let file = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) {
        file += Number(ch);
      } else {
        const color = ch === ch.toUpperCase() ? "w" : "b";
        map.set(FILES[file] + rank, color + ch.toUpperCase());
        file += 1;
      }
    }
  });
  return map;
}

const fileOf = (sq) => FILES.indexOf(sq[0]);
const rankOf = (sq) => Number(sq[1]) - 1;
const distance = (a, b) => Math.hypot(fileOf(a) - fileOf(b), rankOf(a) - rankOf(b));

export class Board {
  constructor(el, { onMove, onPromotion, pieceUrl } = {}) {
    this.el = el;
    this.onMove = onMove || (() => {});
    this.onPromotion = onPromotion || (async () => "q");
    this.pieceUrl = pieceUrl || ((code) => `/static/pieces/${code}.svg`);

    this.orientation = "w";
    this.position = new Map();
    this.pieceEls = new Map();
    this.legal = new Map();
    this.interactive = false;
    this.selected = null;
    this.lastMove = null;
    this.check = null;
    this.arrows = [];
    this.drag = null;

    this.squaresEl = document.createElement("div");
    this.squaresEl.className = "squares";
    this.piecesEl = document.createElement("div");
    this.piecesEl.className = "pieces";
    this.arrowsEl = document.createElementNS(SVG_NS, "svg");
    this.arrowsEl.setAttribute("class", "arrows");
    this.arrowsEl.setAttribute("viewBox", "0 0 100 100");
    this.el.append(this.squaresEl, this.piecesEl, this.arrowsEl);

    this._buildSquares();

    this.el.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    window.addEventListener("pointermove", (e) => this._onPointerMove(e));
    window.addEventListener("pointerup", (e) => this._onPointerUp(e));
    window.addEventListener("pointercancel", () => this._cancelDrag());
    window.addEventListener("resize", () => this._cancelDrag());
  }

  // ------------------------------------------------------------ geometry

  _display(sq) {
    const f = fileOf(sq);
    const r = rankOf(sq);
    return this.orientation === "w" ? { col: f, row: 7 - r } : { col: 7 - f, row: r };
  }

  _squareFromDisplay(col, row) {
    if (col < 0 || col > 7 || row < 0 || row > 7) return null;
    const f = this.orientation === "w" ? col : 7 - col;
    const r = this.orientation === "w" ? 7 - row : row;
    return FILES[f] + (r + 1);
  }

  _squareAt(clientX, clientY) {
    const rect = this.el.getBoundingClientRect();
    const col = Math.floor(((clientX - rect.left) / rect.width) * 8);
    const row = Math.floor(((clientY - rect.top) / rect.height) * 8);
    return this._squareFromDisplay(col, row);
  }

  _place(el, sq) {
    const { col, row } = this._display(sq);
    el.style.transform = `translate(${col * 100}%, ${row * 100}%)`;
  }

  // ------------------------------------------------------------ rendering

  _buildSquares() {
    this.squaresEl.textContent = "";
    this.squareEls = new Map();
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const sq = this._squareFromDisplay(col, row);
        const cell = document.createElement("div");
        const light = (fileOf(sq) + rankOf(sq)) % 2 === 1;
        cell.className = `sq ${light ? "light" : "dark"}`;
        cell.dataset.square = sq;
        if (col === 0) cell.append(this._coord("rank", sq[1]));
        if (row === 7) cell.append(this._coord("file", sq[0]));
        this.squaresEl.append(cell);
        this.squareEls.set(sq, cell);
      }
    }
    this._renderHighlights();
  }

  _coord(kind, text) {
    const span = document.createElement("span");
    span.className = `coord ${kind}`;
    span.textContent = text;
    return span;
  }

  _createPiece(code, sq) {
    const el = document.createElement("div");
    el.className = "piece";
    el.dataset.piece = code;
    el.style.backgroundImage = `url(${this.pieceUrl(code)})`;
    this._place(el, sq);
    this.piecesEl.append(el);
    return el;
  }

  _renderHighlights() {
    const targets = new Set(this.selected ? (this.legal.get(this.selected) || []).map((m) => m.to) : []);
    for (const [sq, cell] of this.squareEls) {
      cell.classList.toggle("last", !!this.lastMove && (sq === this.lastMove.from || sq === this.lastMove.to));
      cell.classList.toggle("selected", sq === this.selected);
      cell.classList.toggle("check", sq === this.check);
      cell.classList.toggle("target", targets.has(sq));
      cell.classList.toggle("occupied", targets.has(sq) && this.position.has(sq));
    }
  }

  _renderArrows() {
    const svg = this.arrowsEl;
    svg.textContent = "";
    const defs = document.createElementNS(SVG_NS, "defs");
    svg.append(defs);

    this.arrows.forEach((a, i) => {
      const id = `arrowhead-${i}`;
      const marker = document.createElementNS(SVG_NS, "marker");
      marker.setAttribute("id", id);
      marker.setAttribute("viewBox", "0 0 10 10");
      marker.setAttribute("refX", "5");
      marker.setAttribute("refY", "5");
      marker.setAttribute("markerWidth", "3");
      marker.setAttribute("markerHeight", "3");
      marker.setAttribute("orient", "auto-start-reverse");
      const tip = document.createElementNS(SVG_NS, "path");
      tip.setAttribute("d", "M0,0 L10,5 L0,10 z");
      tip.setAttribute("fill", a.color);
      marker.append(tip);
      defs.append(marker);

      const from = this._display(a.from);
      const to = this._display(a.to);
      const x1 = from.col * 12.5 + 6.25;
      const y1 = from.row * 12.5 + 6.25;
      let x2 = to.col * 12.5 + 6.25;
      let y2 = to.row * 12.5 + 6.25;
      const len = Math.hypot(x2 - x1, y2 - y1);
      const shorten = 3.2;
      x2 -= ((x2 - x1) / len) * shorten;
      y2 -= ((y2 - y1) / len) * shorten;

      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x2);
      line.setAttribute("y2", y2);
      line.setAttribute("stroke", a.color);
      line.setAttribute("stroke-width", a.width || 2.2);
      line.setAttribute("stroke-linecap", "round");
      line.setAttribute("opacity", a.opacity ?? 0.85);
      line.setAttribute("marker-end", `url(#${id})`);
      svg.append(line);
    });
  }

  // ------------------------------------------------------------ public API

  setOrientation(color) {
    if (color === this.orientation) return;
    this.orientation = color;
    this._buildSquares();
    for (const [sq, el] of this.pieceEls) {
      el.style.transition = "none";
      this._place(el, sq);
    }
    // Re-enable transitions on the next frame so flipping does not animate.
    requestAnimationFrame(() => {
      for (const el of this.pieceEls.values()) el.style.transition = "";
    });
    this._renderArrows();
  }

  flip() {
    this.setOrientation(this.orientation === "w" ? "b" : "w");
  }

  setPosition(fen, { lastMove = null, check = null } = {}) {
    const next = parseFen(fen);
    const old = new Map(this.pieceEls);
    const kept = new Map();
    const appearing = [];

    for (const [sq, code] of next) {
      const el = old.get(sq);
      if (el && el.dataset.piece === code) {
        kept.set(sq, el);
        old.delete(sq);
      } else {
        appearing.push([sq, code]);
      }
    }

    // Pair each appearing piece with the nearest vanished piece of the same
    // kind; that pair is a move and gets animated. Anything left over was
    // captured (fade out) or promoted into existence (fade in).
    const leftovers = [...old.entries()];
    for (const [sq, code] of appearing) {
      let best = -1;
      let bestDist = Infinity;
      leftovers.forEach(([osq, el], i) => {
        if (el && el.dataset.piece === code) {
          const d = distance(osq, sq);
          if (d < bestDist) {
            bestDist = d;
            best = i;
          }
        }
      });
      if (best >= 0) {
        const el = leftovers[best][1];
        leftovers[best][1] = null;
        el.classList.remove("dragging");
        this._place(el, sq);
        kept.set(sq, el);
      } else {
        const el = this._createPiece(code, sq);
        el.classList.add("entering");
        requestAnimationFrame(() => el.classList.remove("entering"));
        kept.set(sq, el);
      }
    }
    for (const [, el] of leftovers) {
      if (!el) continue;
      el.classList.add("fading");
      setTimeout(() => el.remove(), 220);
    }

    this.pieceEls = kept;
    this.position = next;
    this.lastMove = lastMove;
    this.check = check;
    this.selected = null;
    this._renderHighlights();
  }

  // Show a move immediately, before the server confirms it.
  previewMove(uci) {
    const from = uci.slice(0, 2);
    const to = uci.slice(2, 4);
    const promo = uci[4];
    const code = this.position.get(from);
    if (!code) return;
    const next = new Map(this.position);
    next.delete(from);

    // En passant: a pawn moving diagonally onto an empty square.
    if (code[1] === "P" && from[0] !== to[0] && !next.has(to)) {
      next.delete(to[0] + from[1]);
    }
    // Castling: the king moves two files, so the rook jumps over.
    if (code[1] === "K" && Math.abs(fileOf(from) - fileOf(to)) === 2) {
      const rank = from[1];
      const kingside = fileOf(to) > fileOf(from);
      const rookFrom = (kingside ? "h" : "a") + rank;
      const rookTo = (kingside ? "f" : "d") + rank;
      const rook = next.get(rookFrom);
      next.delete(rookFrom);
      if (rook) next.set(rookTo, rook);
    }
    next.set(to, promo ? code[0] + promo.toUpperCase() : code);

    const fen = this._toPlacement(next) + " w - - 0 1";
    this.setPosition(fen, { lastMove: { from, to }, check: null });
  }

  _toPlacement(map) {
    const rows = [];
    for (let rank = 8; rank >= 1; rank--) {
      let row = "";
      let empty = 0;
      for (let f = 0; f < 8; f++) {
        const code = map.get(FILES[f] + rank);
        if (!code) {
          empty++;
          continue;
        }
        if (empty) {
          row += empty;
          empty = 0;
        }
        row += code[0] === "w" ? code[1] : code[1].toLowerCase();
      }
      if (empty) row += empty;
      rows.push(row);
    }
    return rows.join("/");
  }

  setLegalMoves(moves) {
    this.legal = new Map();
    for (const m of moves) {
      if (!this.legal.has(m.from)) this.legal.set(m.from, []);
      this.legal.get(m.from).push(m);
    }
    if (this.selected && !this.legal.has(this.selected)) this.selected = null;
    this._renderHighlights();
  }

  setInteractive(on) {
    this.interactive = on;
    if (!on) this._clearSelection();
  }

  setArrows(arrows) {
    this.arrows = arrows || [];
    this._renderArrows();
  }

  // ------------------------------------------------------------ interaction

  _clearSelection() {
    this.selected = null;
    this._renderHighlights();
  }

  _isTarget(from, to) {
    return (this.legal.get(from) || []).some((m) => m.to === to);
  }

  async _tryMove(from, to) {
    const options = (this.legal.get(from) || []).filter((m) => m.to === to);
    if (!options.length) return false;

    let uci = options[0].uci;
    if (options.some((m) => m.promotion)) {
      const color = (this.position.get(from) || "w")[0];
      const piece = await this.onPromotion(color);
      if (!piece) {
        this.setPosition(this._toPlacement(this.position) + " w - - 0 1", {
          lastMove: this.lastMove,
          check: this.check,
        });
        return false;
      }
      uci = from + to + piece;
    }

    this.interactive = false;
    this.legal = new Map();
    this.selected = null;
    this.previewMove(uci);
    this.onMove(uci);
    return true;
  }

  _onPointerDown(e) {
    if (!this.interactive || e.button > 0) return;
    const sq = this._squareAt(e.clientX, e.clientY);
    if (!sq) return;

    if (this.selected && this._isTarget(this.selected, sq)) {
      e.preventDefault();
      this._tryMove(this.selected, sq);
      return;
    }

    if (this.legal.has(sq)) {
      e.preventDefault();
      const wasSelected = this.selected === sq;
      this.selected = sq;
      this._renderHighlights();
      const el = this.pieceEls.get(sq);
      this.drag = { from: sq, el, x: e.clientX, y: e.clientY, moved: false, wasSelected, hover: null };
    } else {
      this._clearSelection();
    }
  }

  _onPointerMove(e) {
    const d = this.drag;
    if (!d || !d.el) return;
    if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 4) return;
    if (!d.moved) {
      d.moved = true;
      d.el.classList.add("dragging");
    }
    const rect = this.el.getBoundingClientRect();
    const size = rect.width / 8;
    const x = e.clientX - rect.left - size / 2;
    const y = e.clientY - rect.top - size / 2;
    d.el.style.transform = `translate(${x}px, ${y}px)`;

    const sq = this._squareAt(e.clientX, e.clientY);
    if (sq !== d.hover) {
      if (d.hover) this.squareEls.get(d.hover)?.classList.remove("hover");
      d.hover = sq && this._isTarget(d.from, sq) ? sq : null;
      if (d.hover) this.squareEls.get(d.hover)?.classList.add("hover");
    }
  }

  async _onPointerUp(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    if (d.hover) this.squareEls.get(d.hover)?.classList.remove("hover");

    if (!d.moved) {
      if (d.wasSelected) this._clearSelection();
      return;
    }

    const sq = this._squareAt(e.clientX, e.clientY);
    if (sq && sq !== d.from && this._isTarget(d.from, sq)) {
      d.el.classList.remove("dragging");
      d.el.style.transition = "none";
      this._place(d.el, sq);
      requestAnimationFrame(() => (d.el.style.transition = ""));
      const moved = await this._tryMove(d.from, sq);
      if (!moved) this._snapBack(d);
    } else {
      this._snapBack(d);
    }
  }

  _snapBack(d) {
    d.el.classList.remove("dragging");
    this._place(d.el, d.from);
  }

  _cancelDrag() {
    if (this.drag && this.drag.moved) this._snapBack(this.drag);
    this.drag = null;
  }
}
