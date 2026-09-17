// 比較區的 comment 區塊（訓練時寫在 args 裡的 comment）。
const { loadPage, suite, rowByName } = require('./lib');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;
  const RUNS = w.eval('RUNS');
  const listRun = RUNS.find(r => Array.isArray(r.comment) && r.comment.length > 1);
  const strRun = RUNS.find(r => typeof r.comment === 'string' && r.comment.includes(','));
  const noneRun = RUNS.find(r => r.comment == null);
  const chosen = [listRun, strRun, noneRun];
  if (chosen.some(r => !r)) { t.ok('fixture: need list / comma-string / missing comment runs', false); w.close(); return t.result(); }
  chosen.forEach(r => rowByName(d, r.name).querySelector('input[type=checkbox]').click());

  const commentTable = () => {
    const h = [...d.getElementById('cmpbody').querySelectorAll('h3')].find(x => x.textContent === 'comment');
    return h && h.nextElementSibling.nextElementSibling;
  };
  const h3 = [...d.getElementById('cmpbody').querySelectorAll('h3')].map(x => x.textContent);
  t.ok('comment section exists', h3.includes('comment'));
  t.ok('comment section is directly above 相異參數', (h3[h3.indexOf('comment') + 1] || '').startsWith('相異參數'), h3.slice(0, 2).join(' | '));

  const table = commentTable();
  const heads = [...table.querySelectorAll('th')].map(x => x.textContent);
  const tds = [...table.querySelectorAll('tbody td')];
  const cellOf = r => tds[heads.findIndex(h => h.startsWith(r.name.slice(0, 34)))];
  t.ok('one column per selected run', tds.length === 3 && chosen.every(r => cellOf(r)));
  t.ok('columns follow table order like other compare tables',
    JSON.stringify(heads) === JSON.stringify(RUNS.filter(r => chosen.includes(r)).map(r => r.name.slice(0, 34) + '…')));
  t.ok('list comment -> one <li> per item', cellOf(listRun).querySelectorAll('li').length === listRun.comment.length);
  t.ok('list item text intact', cellOf(listRun).querySelector('li').textContent === listRun.comment[0]);
  t.ok('string comment shown whole, not split on commas', cellOf(strRun).querySelectorAll('li').length === 0 && cellOf(strRun).textContent === strRun.comment);
  t.ok('missing comment shows —', cellOf(noneRun).textContent.trim() === '—');

  w.eval('RUNS.find(r => r.id === ' + JSON.stringify(strRun.id) + ').comment = "<img src=x onerror=alert(1)> a<b>"; compare();');
  t.ok('comment HTML is escaped', !commentTable().querySelector('img') && commentTable().innerHTML.includes('&lt;img'));

  w.close();
  return t.result();
};
