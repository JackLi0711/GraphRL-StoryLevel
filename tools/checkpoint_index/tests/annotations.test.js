// 星號 / 附註：合併規則、Filing 後重新對應、篩選、搜尋、寫檔。
// 標註來源全部是記憶體中的 fixture；不會讀寫使用者真實的 annotations.json。
const { loadPage, suite, sleep, rows, rowByName, LS_KEY } = require('./lib');

module.exports = async () => {
  const t = suite();

  // 先載入一次取得資料，用來挑 fixture 的對象
  const probe = loadPage();
  const RUNS = probe.eval('RUNS').map(r => ({ id: r.id, name: r.name, moved: r.moved, recorded: r.recorded_ckpt_dir }));
  probe.close();
  const moved = RUNS.find(r => r.moved && r.recorded && r.recorded.split('/').pop() === r.name);
  const x = RUNS.find(r => r !== moved);
  if (!moved) { t.ok('fixture: need a run that was moved by Filing', false); return t.result(); }

  const OLD = '2026-01-01T00:00:00Z';
  const annotations = { version: 2, annotations: {
    [moved.recorded]: { stars: 2, note: 'file note', updated: OLD },       // 舊路徑，應依資料夾名稱重新對應
    'Results/nope/2020_01_01__gone': { stars: 1, note: 'orphan', updated: OLD },
    [x.id]: { stars: 3, note: '', updated: OLD },                          // 將被較新的 localStorage 清除紀錄覆蓋
  } };
  let written = null, picks = 0;
  const w = loadPage({
    annotations,
    localStorage: { [LS_KEY]: JSON.stringify({ [x.id]: { stars: 0, note: '', updated: '2026-09-01T00:00:00Z' } }) },
    beforeParse(win) {
      win.showSaveFilePicker = async () => {
        picks++;
        return { name: 'annotations.json',
          createWritable: async () => ({ write: async s => { written = s; }, close: async () => {} }) };
      };
    },
  });
  const d = w.document;
  const N = RUNS.length;
  const stars = tr => tr.querySelectorAll('.stars button.on').length;
  const status = () => d.getElementById('annStatus');
  const row = name => rowByName(d, name);
  const chip = label => [...d.querySelectorAll('#starFilter .chip')].find(c => c.textContent === label);

  t.ok('headers include ★ and 附註', [...d.querySelectorAll('#head th')].some(th => th.textContent.startsWith('附註')));
  t.ok('entry under pre-Filing path re-matched to the moved run', stars(row(moved.name)) === 2);
  t.ok('re-matched note is shown', row(moved.name).querySelector('button.note').textContent === 'file note');
  t.ok('newer localStorage clear beats older file entry', stars(row(x.name)) === 0);
  t.ok('status: 1 annotated run, 1 orphan, unsaved', /^1 個 Run 有標註 · 1 筆找不到/.test(status().textContent) && status().classList.contains('bad'), status().textContent);

  // 星號與篩選
  const target = rows(d).map(tr => tr.querySelector('td.name').textContent.slice(0, 22)).find(n => !moved.name.startsWith(n) && !x.name.startsWith(n));
  row(target).querySelectorAll('.stars button')[0].click();
  t.ok('clicking 1st star gives 1★', stars(row(target)) === 1);
  t.ok('star persisted to localStorage', Object.values(JSON.parse(w.localStorage.getItem(LS_KEY))).some(e => e.stars === 1));
  chip('★').click();   t.ok('filter ★ shows only 1-star runs', rows(d).length === 1, rows(d).length);   chip('★').click();
  chip('★★').click();  t.ok('filter ★★ shows the moved run', rows(d).length === 1 && stars(rows(d)[0]) === 2); chip('★★').click();
  chip('無').click();  t.ok('filter 無 shows unstarred runs', rows(d).length === N - 2, rows(d).length);      chip('無').click();
  t.ok('filters cleared', rows(d).length === N);

  // 附註
  row(target).querySelector('button.note').click();
  t.ok('note dialog opens with run name', d.getElementById('noteDlg').open && d.getElementById('noteTitle').textContent.startsWith(target));
  d.getElementById('noteText').value = '重要 baseline <b>x</b>\n第二行';
  d.getElementById('noteDlg').close('save');
  t.ok('saved note shown on button', row(target).querySelector('button.note.has').textContent.startsWith('重要 baseline'));
  const q = d.getElementById('q');
  q.value = '重要 baseline'; q.dispatchEvent(new w.Event('input'));
  t.ok('search matches note text', rows(d).length === 1, rows(d).length);
  q.value = ''; q.dispatchEvent(new w.Event('input'));
  row(target).querySelector('button.note').click();
  d.getElementById('noteText').value = 'SHOULD NOT SAVE';
  d.getElementById('noteDlg').close('cancel');
  t.ok('cancel does not change note', !w.localStorage.getItem(LS_KEY).includes('SHOULD NOT SAVE'));
  row(target).querySelector('button.note').click();
  const ta = d.getElementById('noteText');
  ta.value = '重要 baseline <b>x</b>\n第二行 (ctrl+enter)';
  ta.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'Enter', ctrlKey: true }));
  t.ok('Ctrl+Enter saves', w.localStorage.getItem(LS_KEY).includes('(ctrl+enter)'));

  // 比較區
  row(target).querySelector('input[type=checkbox]').click();
  row(moved.name).querySelector('input[type=checkbox]').click();
  const cmp = d.getElementById('cmpbody').innerHTML;
  t.ok('compare shows ★ row', cmp.includes('<td>★</td>'));
  t.ok('compare escapes note HTML', cmp.includes('&lt;b&gt;x&lt;/b&gt;') && !cmp.includes('<b>x</b>'));

  // 寫檔
  d.getElementById('annSave').click();
  await sleep(50);
  const out = JSON.parse(written);
  const keys = Object.keys(out.annotations);
  t.ok('file written once with version 2', out.version === 2 && picks === 1);
  t.ok('keys sorted (stable git diff)', JSON.stringify(keys) === JSON.stringify([...keys].sort()));
  t.ok('moved run saved under current path, not the old one', keys.includes(moved.id) && !keys.includes(moved.recorded));
  t.ok('orphan preserved', keys.includes('Results/nope/2020_01_01__gone'));
  t.ok('cleared entry dropped from file', !keys.includes(x.id));
  t.ok('new star + note included', Object.values(out.annotations).some(e => e.stars === 1 && e.note.includes('(ctrl+enter)')));
  t.ok('status clean after save', !status().classList.contains('bad'), status().textContent);

  // 自動寫入
  const t10 = rows(d)[10].querySelector('td.name').textContent.slice(0, 22);
  row(t10).querySelectorAll('.stars button')[2].click();
  t.ok('unsaved right after edit', status().classList.contains('bad'));
  await sleep(800);
  t.ok('auto-saved without asking for the file again', picks === 1 && Object.values(JSON.parse(written).annotations).some(e => e.stars === 3));
  t.ok('status clean after auto-save', !status().classList.contains('bad'), status().textContent);
  row(t10).querySelectorAll('.stars button')[2].click();
  t.ok('clicking the same star clears it', stars(row(t10)) === 0);

  [...d.querySelectorAll('#head th')].find(th => th.textContent.startsWith('★')).click();
  t.ok('sorting by ★ puts the most-starred run first', stars(rows(d)[0]) === 2);

  await sleep(800); // 讓最後一次自動寫入的計時器跑完再關閉
  w.close();
  return t.result();
};
