// 開啟頁面時的預設狀態與排序。
const { loadPage, suite, rows } = require('./lib');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;
  const RUNS = w.eval('RUNS');
  const names = () => rows(d).map(tr => tr.querySelector('td.name').textContent);
  // 名稱開頭為 yyyy_mm_dd__hh_mm_ss，字串順序即時間順序；moved 標籤附加在名稱之後不影響前綴
  const nondecreasing = xs => xs.every((x, i) => i === 0 || xs[i - 1] <= x);

  t.ok('指標定義 panel is collapsed by default', d.querySelector('details.defs').open === false);
  t.ok('all runs rendered', rows(d).length === RUNS.length);
  t.ok('default order is chronological, earliest first (same-day runs by time)', nondecreasing(names()));
  const header = label => [...d.querySelectorAll('#head th')].find(th => th.textContent.startsWith(label));
  t.ok('日期 header shows ascending marker', header('日期').textContent.includes('↑'));

  header('日期').click();
  t.ok('clicking 日期 again reverses order', nondecreasing(names().slice().reverse()));

  header('best').click();
  const bests = rows(d).map(tr => +tr.querySelectorAll('td.num')[0].textContent);
  t.ok('clicking best sorts descending', nondecreasing(bests.slice().reverse()));

  w.close();
  return t.result();
};
