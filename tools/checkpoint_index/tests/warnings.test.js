// args 雙來源不一致等警告要顯示在頁面上（spec C2）。
// 真實資料目前沒有警告，因此在載入前替內嵌資料的第一個 Run 注入一筆。
const { loadPage, suite, rows, rowByName } = require('./lib');

module.exports = async () => {
  const t = suite();

  const clean = loadPage();
  const realWarned = clean.eval('RUNS').filter(r => r.warnings && r.warnings.length).length;
  if (realWarned === 0) {
    t.ok('no warnings in data -> header has no warning count', !clean.document.getElementById('count').textContent.includes('⚠'));
    t.ok('no warnings in data -> no ⚠ tags', clean.document.querySelectorAll('#body .tag.warn').length === 0);
  }
  clean.close();

  const MSG = 'args 來源不一致: lr json=0.001 log=<0.0005>';
  const w = loadPage({
    transformHtml: html => {
      const marker = '"warnings": []';
      if (!html.includes(marker)) throw new Error('embedded data has no empty warnings list to inject into');
      return html.replace(marker, () => '"warnings": ' + JSON.stringify([MSG]));
    },
  });
  const d = w.document;
  const RUNS = w.eval('RUNS');
  const target = RUNS.find(r => r.warnings.length === 1 && r.warnings[0] === MSG);
  const expectedWarned = realWarned + 1;

  t.ok('header shows how many runs have warnings',
    d.getElementById('count').textContent.includes('⚠ ' + expectedWarned + ' 個 Run 有警告'), d.getElementById('count').textContent);
  const tag = rowByName(d, target.name).querySelector('.tag.warn');
  t.ok('warned run shows a ⚠ tag in the table', !!tag && tag.textContent === '⚠ 1');
  t.ok('⚠ tag tooltip lists the warning text', !!tag && tag.title === MSG);
  t.ok('only warned runs have a ⚠ tag', rows(d).filter(tr => tr.querySelector('.tag.warn')).length === expectedWarned);

  // 比較區
  const other = RUNS.filter(r => r !== target && !r.warnings.length);
  rowByName(d, target.name).querySelector('input[type=checkbox]').click();
  rowByName(d, other[0].name).querySelector('input[type=checkbox]').click();
  const body = d.getElementById('cmpbody');
  const h3 = [...body.querySelectorAll('h3')].map(h => h.textContent);
  t.ok('compare shows a 警告 section', h3.some(h => h.includes('警告')));
  t.ok('警告 section comes before comment', h3.findIndex(h => h.includes('警告')) < h3.indexOf('comment'));
  t.ok('comment is still directly above 相異參數', (h3[h3.indexOf('comment') + 1] || '').startsWith('相異參數'));
  t.ok('warning text shown and HTML-escaped', body.innerHTML.includes('log=&lt;0.0005&gt;') && body.textContent.includes(MSG));

  rowByName(d, target.name).querySelector('input[type=checkbox]').click();
  rowByName(d, other[1].name).querySelector('input[type=checkbox]').click();
  t.ok('no 警告 section when none of the chosen runs has warnings',
    ![...d.getElementById('cmpbody').querySelectorAll('h3')].some(h => h.textContent.includes('警告')));

  w.close();
  return t.result();
};
