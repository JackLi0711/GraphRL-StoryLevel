// Rejected Step：標示檢核項目、載重組合、實際值與限值；之後回到最終設計（最後一個通過的設計）。
const { loadPage, suite, status, click, fixtures } = require('./lib');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;
  const doc = fixtures.dqn;
  const n = doc.steps.length;
  const slider = d.getElementById('slider');
  const goTo = i => { slider.value = String(i); slider.dispatchEvent(new w.Event('input', { bubbles: true })); };

  goTo(n);  // the Rejected Step
  const s = status(d);
  t.ok('says Rejected', s.includes('Rejected'), s);
  t.ok('names the check', s.includes('Story drift ratio'), s);
  t.ok('names the load case', s.includes('EQX+'));
  t.ok('shows value vs limit', s.includes('0.00613 > limit 0.005'), s);
  t.ok('status panel flagged as rejected', d.getElementById('status').classList.contains('rejected'));
  const g = doc.steps[n - 1].action;
  t.ok('rejected reduction drawn with the rejected halo', d.querySelectorAll('#frame line.halo.rejected').length === doc.topology.members.filter(m => m[2] === g).length);
  t.ok('heatmap chosen cell flagged rejected', !!d.querySelector('#heat .cell.chosen.rejected'));
  t.ok('curve marks the rejected step', !!d.querySelector('#curve .rejmark'));

  click(d, 'next');  // rolled-back final design
  const f = status(d);
  t.ok('then shows the final design', f.includes('Final design'), f);
  t.ok('final Saving Ratio matches the document', f.includes((doc.end.saving_ratio * 100).toFixed(1) + '%'), f);
  const shown = d.querySelector(`#frame line.m[data-group="${g}"]`);
  t.ok('the rejected group is back at its previous section', shown.classList.contains('s7'), shown.getAttribute('class'));
  t.ok('no chosen group on the final design', d.querySelectorAll('#frame line.halo').length === 0);
  return t.result();
};
