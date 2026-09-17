// 內嵌資料的不變量：檢視器畫面所依賴的前提。
const { loadPage, suite } = require('./lib');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const RUNS = w.eval('RUNS');
  const F = w.eval('FIELDS');

  for (const r of RUNS) {
    const m = r.metrics;
    if (!m || typeof m.best !== 'number') { t.ok('metrics present: ' + r.name, false); continue; }
    if (m.final_design.length !== 4 * m.story_num) t.ok('design length = 4 x story_num: ' + r.name, false);
    if (m.initial_design.length !== m.final_design.length) t.ok('initial/final design same length: ' + r.name, false);
    if (Math.abs((m.best - m.last) - m.gap) > 1e-9) t.ok('gap = best - last: ' + r.name, false, m.gap);
    if (m.unconverged !== (m.gap > 0.1)) t.ok('unconverged flag matches gap > 0.1: ' + r.name, false);
  }
  t.ok('every run has metrics, consistent design length, gap and unconverged flag', t.result().fail === 0, RUNS.length + ' runs');

  // 每個 args key 都恰好落在一個欄位層級，否則比較表會出現空白的層級標籤
  const tiers = [F.universal, F.invariant, ...Object.values(F.algorithm_specific)];
  const allKeys = new Set(RUNS.flatMap(r => Object.keys(r.args)));
  const inTiers = k => tiers.filter(ts => ts.includes(k)).length;
  const bad = [...allKeys].filter(k => inTiers(k) !== 1);
  t.ok('each arg key is in exactly one field tier', bad.length === 0, bad.join(', ') || allKeys.size + ' keys');

  for (const ax of ['algorithm', 'purpose', 'operator', 'device']) {
    t.ok('no null ' + ax, RUNS.every(r => r[ax] != null));
  }
  // Operator = 地點-人（CONTEXT.md: Operator）；目前資料的機器都認得，不應出現 Unknown- 或無法推定的人
  t.ok('every operator is Location-Person', RUNS.every(r => /^(Server|Local|Unknown)-\w+$/.test(r.operator)),
    [...new Set(RUNS.map(r => r.operator))].join(', '));
  t.ok('no run has an unresolved operator on current data', RUNS.every(r => !r.operator.startsWith('Unknown-') && !r.operator.endsWith('-unknown')));
  t.ok('every run has a yyyy_mm_dd date', RUNS.every(r => /^\d{4}_\d{2}_\d{2}$/.test(r.date || '')));
  t.ok('run ids are unique', new Set(RUNS.map(r => r.id)).size === RUNS.length);
  t.ok('run folder names are unique (annotations re-match on this after Filing)',
    new Set(RUNS.map(r => r.id.split('/').pop())).size === RUNS.length);

  w.close();
  return t.result();
};
