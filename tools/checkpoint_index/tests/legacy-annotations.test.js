// 舊格式相容：version 1 的 annotations.json 與舊 localStorage 把附註存在 comment 欄位。
// 讀取時要能正確顯示；寫出時一律轉為 version 2 的 note 欄位，不再出現 comment。
const { loadPage, suite, sleep, rowByName, LS_KEY } = require('./lib');

module.exports = async () => {
  const t = suite();
  const probe = loadPage();
  const [a, b] = probe.eval('RUNS').slice(0, 2).map(r => ({ id: r.id, name: r.name }));
  probe.close();

  let written = null;
  const w = loadPage({
    annotations: { version: 1, annotations: { [a.id]: { stars: 2, comment: 'v1 file note', updated: '2026-01-01T00:00:00Z' } } },
    localStorage: { [LS_KEY]: JSON.stringify({ [b.id]: { stars: 1, comment: 'v1 local note', updated: '2026-01-02T00:00:00Z' } }) },
    beforeParse(win) {
      win.showSaveFilePicker = async () => ({ name: 'annotations.json',
        createWritable: async () => ({ write: async s => { written = s; }, close: async () => {} }) });
    },
  });
  const d = w.document;
  const noteText = name => rowByName(d, name).querySelector('button.note').textContent;

  t.ok('v1 file entry (comment) is shown as a note', noteText(a.name) === 'v1 file note');
  t.ok('v1 localStorage entry (comment) is shown as a note', noteText(b.name) === 'v1 local note');
  t.ok('v1 note is searchable', (() => {
    const q = d.getElementById('q'); q.value = 'v1 local'; q.dispatchEvent(new w.Event('input'));
    const n = d.querySelectorAll('#body tr').length; q.value = ''; q.dispatchEvent(new w.Event('input'));
    return n === 1;
  })());

  rowByName(d, a.name).querySelector('button.note').click();
  t.ok('editing dialog is prefilled from the legacy field', d.getElementById('noteText').value === 'v1 file note');
  d.getElementById('noteDlg').close('cancel');

  d.getElementById('annSave').click();
  await sleep(50);
  const out = JSON.parse(written);
  const entries = Object.values(out.annotations);
  t.ok('saved file is version 2', out.version === 2);
  t.ok('saved entries use note, never comment', entries.length === 2 && entries.every(e => 'note' in e && !('comment' in e)));
  t.ok('legacy note text preserved on save', out.annotations[a.id].note === 'v1 file note' && out.annotations[b.id].note === 'v1 local note');
  t.ok('stars and updated preserved on save', out.annotations[a.id].stars === 2 && out.annotations[a.id].updated === '2026-01-01T00:00:00Z');

  w.close();
  return t.result();
};
