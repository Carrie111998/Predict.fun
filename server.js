// predict.fun 5-minute Up/Down PREDICTOR for BTC, ETH, BNB.  Prediction only - places no orders.
// Run:  node server.js     then open  http://localhost:8790
//
// For every coin it shows:
//   * CURRENT round: live probability of Up (price vs round open, scaled by recent volatility), the
//     Predict.fun price for each side, and a plain suggestion: which side, the most you should pay,
//     and the stake that returns your target profit (default $0.25).
//   * NEXT round: market price + a momentum lean.  Honest note: the next round is close to a coin flip.
//   * A live scoreboard: every prediction at 1:00 / 2:00 / 3:00 / 4:00 into each round is logged and
//     checked against the result, so you can see how accurate each checkpoint really is.
// Zero dependencies (Node 18+). Uses PREDICT_API_KEY from .env for Predict.fun order books.
const http = require('http'), fs = require('fs'), path = require('path');

for (const l of (fs.existsSync(path.join(__dirname, '.env')) ? fs.readFileSync(path.join(__dirname, '.env'), 'utf8') : '').split('\n')) {
  const m = l.trim().match(/^([A-Z_]+)=(.*)$/); if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim();
}
const PORT = +process.env.PREDICTOR_PORT || 8790, KEY = process.env.PREDICT_API_KEY;
const SYMS = { btc: 'BTCUSDT', eth: 'ETHUSDT', bnb: 'BNBUSDT' }, R = 300000, M = 60000;
const DATA = path.join(__dirname, 'data'), LOG = path.join(DATA, 'predictions.jsonl');
fs.mkdirSync(DATA, { recursive: true });

// ---------------- data sources ----------------
const BINANCE = ['https://api.binance.com', 'https://data-api.binance.vision'];
async function bj(p) { for (const b of BINANCE) { try { const r = await fetch(b + p, { signal: AbortSignal.timeout(8000) }); if (r.ok) return r.json(); } catch {} } throw new Error('Binance unreachable'); }
async function pf(p) {
  const r = await fetch('https://api.predict.fun' + p, { headers: { 'x-api-key': KEY }, signal: AbortSignal.timeout(8000) });
  if (!r.ok) throw new Error(p + ' ' + r.status); return (await r.json()).data;
}
const slugOf = (k, start) => `${k}-updown-5m-${start / 1000}`;

const state = {}; // per coin: candles (1m, 2 days), price
for (const k of Object.keys(SYMS)) state[k] = { m1: [], price: null };
const books = {};   // slug -> { bid, ask, t }  (Up side; Down = 1 - Up)
const marketIds = {};

async function loadCandles(k, full) {
  const s = state[k], now = Date.now();
  let from = full || !s.m1.length ? now - 2 * 864e5 : s.m1[s.m1.length - 1].t;
  const out = [];
  while (from < now) {
    const kl = await bj(`/api/v3/klines?symbol=${SYMS[k]}&interval=1m&startTime=${from}&limit=1000`);
    if (!kl.length) break; out.push(...kl); from = kl[kl.length - 1][0] + M; if (kl.length < 1000) break;
  }
  const map = new Map(s.m1.map(x => [x.t, x]));
  for (const x of out) map.set(x[0], { t: x[0], o: +x[1], h: +x[2], l: +x[3], c: +x[4], v: +x[5] });
  s.m1 = [...map.values()].sort((a, b) => a.t - b.t).filter(x => x.t >= now - 2 * 864e5);
}
async function refreshPrices() {
  try {
    const arr = await bj('/api/v3/ticker/price?symbols=' + encodeURIComponent(JSON.stringify(Object.values(SYMS))));
    for (const [k, sym] of Object.entries(SYMS)) state[k].price = +arr.find(x => x.symbol === sym).price;
  } catch (e) { console.error('price', e.message); }
}
async function refreshBooks() {
  if (!KEY) return;
  const cur = Math.floor(Date.now() / R) * R;
  await Promise.all(Object.keys(SYMS).flatMap(k => [cur, cur + R].map(async st => {
    const slug = slugOf(k, st);
    try {
      if (!marketIds[slug]) marketIds[slug] = (await pf('/v1/categories/' + slug)).markets[0].id;
      const ob = await pf('/v1/markets/' + marketIds[slug] + '/orderbook');
      const bid = ob.bids.length ? Math.max(...ob.bids.map(x => x[0])) : null, ask = ob.asks.length ? Math.min(...ob.asks.map(x => x[0])) : null;
      books[slug] = { bid, ask, t: Date.now() };
    } catch (e) { if (!/404/.test(e.message)) console.error('book', slug, e.message); }
  })));
}

// ---------------- the model ----------------
const erf = x => { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); };
const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
// per-minute volatility: EWMA of 1m log returns over the last 90 minutes
function sigmaPerMin(m1) { const c = m1.slice(-91).map(x => x.c); let v = 0; const a = 2 / 31;
  for (let i = 1; i < c.length; i++) { const r = Math.log(c[i] / c[i - 1]); v = a * r * r + (1 - a) * v; } return Math.sqrt(v); }
// probability that the round closes above its open, given price now and minutes left
const pUpNow = (price, open, sig, minLeft) => (open && price && sig > 0) ? Phi(Math.log(price / open) / (sig * Math.sqrt(Math.max(minLeft, 0.02)))) : null;
const emaLast = (arr, n) => { const k = 2 / (n + 1); let e = arr[0]; for (const x of arr) e = x * k + e * (1 - k); return e; };

// ---------------- calibration: how often is the model right at each confidence level? ----------------
// Replays the last 2 days of 5-minute rounds at 1:00, 2:00, 3:00 and 4:00 using the same model, buckets the
// stated confidence, and measures the real hit rate. The tool then uses the REAL hit rate, not the raw number.
const BUCKETS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001];
let calib = { table: [], at: 0 };
function buildCalibration() {
  const cnt = BUCKETS.slice(0, -1).map(() => ({ n: 0, hit: 0 }));
  for (const k of Object.keys(SYMS)) {
    const m1 = state[k].m1, idx = new Map(m1.map((x, i) => [x.t, i]));
    for (let i = 95; i < m1.length; i++) {
      const st = m1[i].t; if (st % R !== 0) continue;
      const iEnd = idx.get(st + 4 * M); if (iEnd == null) continue;
      const open = m1[i].o, res = Math.sign(m1[iEnd].c - open); if (!res) continue;
      for (const cp of [1, 2, 3, 4]) {
        const j = idx.get(st + (cp - 1) * M); if (j == null) continue;
        const sig = sigmaPerMin(m1.slice(0, i)), p = pUpNow(m1[j].c, open, sig, 5 - cp); if (p == null) continue;
        const side = p >= 0.5 ? 1 : -1, conf = side === 1 ? p : 1 - p;
        const b = BUCKETS.findIndex((x, q) => conf >= x && conf < BUCKETS[q + 1]); if (b < 0) continue;
        cnt[b].n++; if (side === res) cnt[b].hit++;
      }
    }
  }
  calib = { at: Date.now(), table: cnt.map((c, q) => ({ from: BUCKETS[q], to: Math.min(1, BUCKETS[q + 1]), n: c.n, real: c.n ? c.hit / c.n : null })) };
}
// calibrated probability for a raw model confidence (Laplace-smoothed, never above the raw value)
function calibrated(conf) {
  const b = calib.table.find(r => conf >= r.from && conf < (r.to === 1 ? 1.0001 : r.to));
  if (!b || !b.n) return conf;
  return Math.min(conf, (b.n * b.real + 2 * 0.5) / (b.n + 2));
}

// ---------------- prediction log + scoreboard ----------------
const logged = new Set(); // slug|checkpoint
let scoreCache = null, scoreAt = 0;
function logPrediction(rec) { fs.appendFileSync(LOG, JSON.stringify(rec) + '\n'); }
function scoreboard() {
  if (Date.now() - scoreAt < 15000 && scoreCache) return scoreCache;
  const lines = fs.existsSync(LOG) ? fs.readFileSync(LOG, 'utf8').split('\n').filter(Boolean).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean) : [];
  const byCp = {};
  for (const p of lines) {
    const st = +p.slug.split('-').pop() * 1000, m1 = state[p.sym]?.m1 || [];
    const first = m1.find(x => x.t === st), last = m1.find(x => x.t === st + 4 * M);
    if (!first || !last || st + R > Date.now() - 20000) continue;
    const res = Math.sign(last.c - first.o); if (!res) continue;
    for (const key of p.buy ? [p.cp, 'BUY suggestions'] : [p.cp]) {
    const b = byCp[key] ||= { n: 0, right: 0, priced: 0, pnl: 0, staked: 0 };
    b.n++; const right = (p.side === 'UP' ? 1 : -1) === res; if (right) b.right++;
    if (p.cost && p.cost > 0 && p.cost < 100) { b.priced++; b.staked += 1; b.pnl += right ? (100 - p.cost) / p.cost : -1; }
    }
  }
  scoreCache = Object.entries(byCp).sort((a, b) => a[0].localeCompare(b[0])).map(([cp, b]) => ({ cp, n: b.n, accuracy: b.n ? b.right / b.n : null, priced: b.priced, roiPerDollar: b.priced ? b.pnl / b.priced : null }));
  scoreAt = Date.now(); return scoreCache;
}

// ---------------- next-round lean: momentum of last 15 minutes + how often "follow last round" worked ----------------
function nextRoundLean(k) {
  const m1 = state[k].m1; if (m1.length < 60) return null;
  const closes = m1.slice(-30).map(x => x.c), ema5 = emaLast(closes.slice(-10), 5), ema15 = emaLast(closes, 15);
  const lean = ema5 >= ema15 ? 'UP' : 'DOWN';
  // historical check over the last 2 days of 5-minute rounds: does the round follow the 15-min EMA lean?
  let n = 0, hit = 0;
  for (let i = 60; i + 5 < m1.length; i++) {
    if (m1[i].t % R !== 0) continue;
    const c = m1.slice(i - 30, i).map(x => x.c), l = emaLast(c.slice(-10), 5) >= emaLast(c, 15) ? 1 : -1;
    const res = Math.sign(m1[i + 4].c - m1[i].o); if (!res) continue; n++; if (l === res) hit++;
  }
  return { lean, histAccuracy: n ? hit / n : null, rounds: n };
}

// ---------------- the API payload ----------------
let TARGET = 0.25; // target profit per winning bet in USD
function predict() {
  const now = Date.now(), cur = Math.floor(now / R) * R, el = (now - cur) / M, minLeft = 5 - el;
  const out = { now, roundStart: cur, roundEnd: cur + R, elapsedMin: el, target: TARGET, coins: {} };
  for (const k of Object.keys(SYMS)) {
    const s = state[k], first = s.m1.find(x => x.t === cur), open = first ? first.o : null, price = s.price;
    const sig = sigmaPerMin(s.m1), pUp = pUpNow(price, open, sig, minLeft);
    const book = books[slugOf(k, cur)], nbook = books[slugOf(k, cur + R)];
    const upAsk = book?.ask ?? null, dnAsk = book?.bid != null ? +(1 - book.bid).toFixed(4) : null;
    let current = { open, price, movePct: open && price ? (price / open - 1) * 100 : null, pUp, upAsk, dnAsk };
    if (pUp != null) {
      const side = pUp >= 0.5 ? 'UP' : 'DOWN', raw = side === 'UP' ? pUp : 1 - pUp, p = calibrated(raw), ask = side === 'UP' ? upAsk : dnAsk;
      // most you should pay: model probability minus a 3-point safety margin (fees, slippage, model error)
      const maxPay = Math.max(0, p - 0.03), worth = ask != null && ask <= maxPay && ask < 0.97;
      const stakeFor = ask ? TARGET * ask / (1 - ask) : null; // stake whose win pays TARGET profit
      let verdict;
      if (el < 0.5) verdict = 'Too early – round just opened, the price has not moved yet';
      else if (ask == null) verdict = book ? side + ' has no sellers left (already ~99¢) – round is decided, skip' : 'No market price yet';
      else if (worth) verdict = `BUY ${side} at ${Math.round(ask * 100)}¢ or less – stake $${stakeFor.toFixed(2)} to win about $${TARGET.toFixed(2)}`;
      else if (ask >= 0.97) verdict = `${side} is already ${Math.round(ask * 100)}¢ – profit too small, skip`;
      else verdict = `Lean ${side}, but ${Math.round(ask * 100)}¢ is more than it is worth (${Math.round(maxPay * 100)}¢) – skip or wait for a dip`;
      current = { ...current, side, confidence: p, rawConfidence: raw, ask, maxPay, worth, stakeFor, verdict };
      // log the prediction once at each checkpoint (1:00, 2:00, 3:00, 4:00)
      for (const cp of [1, 2, 3, 4]) {
        const key = slugOf(k, cur) + '|' + cp;
        if (el >= cp && el < cp + 0.25 && !logged.has(key)) {
          logged.add(key); logPrediction({ t: now, slug: slugOf(k, cur), sym: k, cp: cp + ':00', side, confidence: +p.toFixed(3), cost: ask != null ? +(ask * 100).toFixed(1) : null, movePct: current.movePct, buy: !!worth });
        }
      }
    }
    const lean = nextRoundLean(k);
    const next = { upAsk: nbook?.ask ?? null, dnAsk: nbook?.bid != null ? +(1 - nbook.bid).toFixed(4) : null, lean: lean?.lean, histAccuracy: lean?.histAccuracy, rounds: lean?.rounds };
    out.coins[k] = { current, next, sigmaPerMinPct: sig * 100, last12: s.m1.filter(x => x.t % R === 0 && x.t < cur).slice(-12).map(x => { const e = s.m1.find(y => y.t === x.t + 4 * M); return e ? Math.sign(e.c - x.o) : 0; }) };
  }
  out.scoreboard = scoreboard();
  out.calibration = calib.table;
  return out;
}

// ---------------- HTTP ----------------
const send = (res, code, body, type = 'application/json') => { res.writeHead(code, { 'content-type': type, 'cache-control': 'no-store' }); res.end(typeof body === 'string' ? body : JSON.stringify(body)); };
const handler = async (req, res) => {
  const u = new URL(req.url, 'http://x');
  try {
    if (u.pathname === '/api/predict') return send(res, 200, predict());
    if (u.pathname === '/api/target' && req.method === 'POST') {
      let b = ''; req.on('data', d => (b += d)); await new Promise(r => req.on('end', r));
      const v = +JSON.parse(b || '{}').target; if (v >= 0.05 && v <= 5) TARGET = v; return send(res, 200, { target: TARGET });
    }
    if (u.pathname === '/' || u.pathname === '/index.html') return send(res, 200, fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8'), 'text/html; charset=utf-8');
    send(res, 404, { error: 'not found' });
  } catch (e) { send(res, 500, { error: e.message }); }
};
// Listen on this PC and on the Tailscale address (100.64.0.0/10) so a phone on Tailscale can open it; never the LAN.
const hosts = ['127.0.0.1', ...Object.values(require('os').networkInterfaces()).flat()
  .filter(i => i.family === 'IPv4' && /^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\./.test(i.address)).map(i => i.address)];
for (const h of hosts) {
  const srv = http.createServer(handler);
  srv.on('error', e => console.error('listen', h, e.message));
  srv.listen(PORT, h, () => console.log('Predictor: http://' + (h === '127.0.0.1' ? 'localhost' : h) + ':' + PORT));
}

(async () => {
  console.log(KEY ? 'Predict.fun API key loaded' : 'No PREDICT_API_KEY in .env: market prices will be missing');
  await Promise.all(Object.keys(SYMS).map(k => loadCandles(k, true).catch(e => console.error('candles', k, e.message))));
  await refreshPrices(); await refreshBooks();
  console.log('Loaded:', Object.entries(state).map(([k, s]) => `${k} ${s.m1.length} candles`).join(', '));
  buildCalibration(); console.log('Calibration:', calib.table.map(r => `${Math.round(r.from * 100)}-${Math.round(r.to * 100)}%: real ${r.real == null ? '-' : Math.round(r.real * 100) + '%'} (${r.n})`).join(' | '));
  setInterval(buildCalibration, 600000);
  setInterval(refreshPrices, 2000);
  setInterval(refreshBooks, 3000);
  setInterval(() => Object.keys(SYMS).forEach(k => loadCandles(k).catch(() => {})), 10000);
})();
