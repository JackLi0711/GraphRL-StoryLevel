// 日期篩選：區間 AND 年 AND 月 AND 日。以頁面上真正的 matchDate 執行。
const { loadPage, suite, rows } = require('./lib');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;
  const RUNS = w.eval('RUNS');
  const count = setup => {
    w.eval('DATE.from=""; DATE.to=""; DATE.year.clear(); DATE.month.clear(); DATE.day.clear();');
    w.eval(setup);
    return RUNS.filter(r => w.eval('matchDate')(r)).length;
  };
  const iso = r => r.date.replace(/_/g, '-');
  const part = (r, i) => r.date.split('_')[i];
  const ref = f => RUNS.filter(f).length;
  const firstYear = part(RUNS[0], 0);
  const lastIso = RUNS.map(iso).sort().pop();

  t.ok('no filter = all', count('') === RUNS.length);
  t.ok('single year', count(`DATE.year.add("${firstYear}")`) === ref(r => part(r, 0) === firstYear));
  t.ok('all years = all', count([...new Set(RUNS.map(r => part(r, 0)))].map(y => `DATE.year.add("${y}")`).join(';')) === RUNS.length);
  t.ok('month across years', count('DATE.month.add("04")') === ref(r => part(r, 1) === '04'));
  t.ok('two months = OR', count('DATE.month.add("04");DATE.month.add("05")') === ref(r => ['04', '05'].includes(part(r, 1))));
  t.ok('year AND month', count('DATE.year.add("2026");DATE.month.add("04")') === ref(r => r.date.startsWith('2026_04')));
  t.ok('day', count('DATE.day.add("11")') === ref(r => part(r, 2) === '11'));
  t.ok('range inclusive on both ends', count('DATE.from="2026-04-01";DATE.to="2026-04-30"') === ref(r => iso(r) >= '2026-04-01' && iso(r) <= '2026-04-30'));
  t.ok('single-day range includes that day', count(`DATE.from="${lastIso}";DATE.to="${lastIso}"`) === ref(r => iso(r) === lastIso));
  t.ok('from only', count('DATE.from="2026-01-01"') === ref(r => iso(r) >= '2026-01-01'));
  t.ok('to only', count('DATE.to="2025-12-31"') === ref(r => iso(r) <= '2025-12-31'));
  t.ok('inverted range = 0', count('DATE.from="2026-05-01";DATE.to="2026-04-01"') === 0);
  t.ok('range AND month', count('DATE.from="2026-01-01";DATE.to="2026-12-31";DATE.month.add("03")') === ref(r => r.date.startsWith('2026_03')));

  // 透過 UI：勾選年份下拉選單
  w.eval('DATE.from=""; DATE.to=""; DATE.year.clear(); DATE.month.clear(); DATE.day.clear(); render();');
  const yearMenu = d.querySelectorAll('#dParts details.dd')[0];
  const cb = [...yearMenu.querySelectorAll('label')].find(l => l.textContent.startsWith(firstYear)).querySelector('input');
  cb.click();
  t.ok('ticking a year in the dropdown filters the table', rows(d).length === ref(r => part(r, 0) === firstYear), rows(d).length);
  t.ok('dropdown label reflects selection', yearMenu.querySelector('summary').textContent.includes(firstYear));
  const from = d.getElementById('dFrom'), to = d.getElementById('dTo');
  from.value = '2026-05-01'; from.dispatchEvent(new w.Event('change'));
  to.value = '2026-04-01'; to.dispatchEvent(new w.Event('change'));
  t.ok('inverted range shows a warning', d.getElementById('dWarn').textContent.length > 0);
  d.getElementById('dClear').click();
  t.ok('清除日期 resets filter and checkboxes', rows(d).length === RUNS.length && !cb.checked && !d.getElementById('dWarn').textContent);

  w.close();
  return t.result();
};
