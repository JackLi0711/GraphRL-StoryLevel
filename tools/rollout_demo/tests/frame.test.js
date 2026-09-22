// 結構圖：構件依斷面大小上色、所選群組加外框、hover 顯示群組、斷面與 Action Preference。
const { loadPage, suite, click, fixtures } = require('./lib');

const secClass = el => [...el.classList].find(c => /^s\d$/.test(c));

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;
  const doc = fixtures.dqn;
  const inGroup = g => doc.topology.members.filter(m => m[2] === g).length;

  const members = d.querySelectorAll('#frame line.m');
  t.ok('one line per member', members.length === doc.topology.members.length, members.length);
  t.ok('no chosen group on the initial design', d.querySelectorAll('#frame line.m.chosen, #frame line.halo').length === 0);
  const g0 = d.querySelector('#frame line.m[data-group="0"]');
  const g1 = d.querySelector('#frame line.m[data-group="1"]');
  t.ok('smaller section is drawn with a lighter step than the largest', secClass(g0) === 's1' && secClass(g1) === 's7', secClass(g0) + ' ' + secClass(g1));

  click(d, 'next');
  const action = doc.steps[0].action;
  t.ok('chosen group members are marked', d.querySelectorAll('#frame line.m.chosen').length === inGroup(action));
  t.ok('chosen group has a halo per member', d.querySelectorAll('#frame line.halo:not(.rejected)').length === inGroup(action));
  t.ok('chosen members recolored after the reduction', secClass(d.querySelector(`#frame line.m[data-group="${action}"]`)) === 's1');

  // hover：以 mousemove 觸發 tooltip
  click(d, 'next');
  const hit = d.querySelector('#frame line.hit[data-group="3"]');
  hit.dispatchEvent(new w.MouseEvent('mousemove', { bubbles: true, clientX: 10, clientY: 10 }));
  const tip = d.getElementById('tip');
  t.ok('tooltip visible on hover', tip.style.display === 'block');
  t.ok('tooltip names the group', tip.textContent.includes('4F X-dir beam'), tip.textContent);
  t.ok('tooltip shows the section designation', tip.textContent.includes(doc.section_catalog.beam[14]));
  t.ok('tooltip shows the DQN preference as a Q value', tip.textContent.includes('Q value: ' + doc.steps[1].preference[3].toFixed(3)), tip.textContent);
  t.ok('hovered group is highlighted in the heatmap too', d.querySelector('#heat .cell.hl').dataset.group === '3');
  click(d, 'next');  // Design Step 3: group 0 reached its minimum section in Design Step 2
  const na = d.querySelector('#frame line.hit[data-group="0"]');
  na.dispatchEvent(new w.MouseEvent('mousemove', { bubbles: true, clientX: 10, clientY: 10 }));
  t.ok('tooltip says a minimum-section group is not selectable', tip.textContent.includes('Not selectable'), tip.textContent);
  d.getElementById('frame').dispatchEvent(new w.MouseEvent('mouseleave'));
  t.ok('tooltip hides on leave', tip.style.display === 'none');
  return t.result();
};
