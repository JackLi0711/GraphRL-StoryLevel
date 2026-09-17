// 比較區的呈現：設計格子說明必須同時顯示 best / last / gap（ADR-0001），
// 所有 Run 名稱都要經過 HTML 跳脫。
const { loadPage, suite, rowByName } = require('./lib');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;
  const RUNS = w.eval('RUNS');
  const unconv = RUNS.find(r => r.metrics.unconverged);
  const conv = RUNS.find(r => !r.metrics.unconverged);
  [conv, unconv].forEach(r => rowByName(d, r.name).querySelector('input[type=checkbox]').click());

  const cards = [...d.querySelectorAll('#cmpbody .gcard')];
  t.ok('one design card per chosen run', cards.length === 2);
  const legendOf = r => cards.find(c => c.querySelector('h3').textContent.startsWith(r.name.slice(0, 40))).querySelector('.legend');
  for (const r of [conv, unconv]) {
    const text = legendOf(r).textContent;
    t.ok('legend shows best, last and gap together (' + (r.metrics.unconverged ? 'unconverged' : 'converged') + ')',
      text.includes('best ' + (r.metrics.best * 100).toFixed(1) + '%')
      && text.includes('last ' + (r.metrics.last * 100).toFixed(1) + '%')
      && text.includes('gap ' + r.metrics.gap.toFixed(3)), text);
  }
  t.ok('unconverged run legend flags gap in red', !!legendOf(unconv).querySelector('.bad') && legendOf(unconv).textContent.includes('未收斂'));
  t.ok('converged run legend has no red gap', !legendOf(conv).querySelector('.bad'));

  // 名稱跳脫：把兩個 Run 的名稱改成含 HTML 的字串後重新繪製比較區
  w.eval('RUNS.filter(r => sel.has(r.id)).forEach((r, i) => { r.name = "<b class=injected>x" + i + "</b>" + r.name; }); compare();');
  const body = d.getElementById('cmpbody');
  t.ok('no injected element in any compare header or card title', body.querySelectorAll('.injected').length === 0);
  const tables = [...body.querySelectorAll('table.diff')];
  t.ok('every compare table header shows the escaped name',
    tables.length >= 3 && tables.every(tb => [...tb.querySelectorAll('th')].some(th => th.textContent.startsWith('<b class=injected>'))),
    tables.length + ' tables');

  w.close();
  return t.result();
};
