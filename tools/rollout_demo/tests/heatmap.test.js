// Action Preference 熱圖：6 層 × 4 類、標示 DQN=Q value / PPO=action probability、不可選者標為不可用、所選格標示。
const { loadPage, suite, click, runButton, fixtures } = require('./lib');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;
  const cells = () => [...d.querySelectorAll('#heat .cell')];

  t.ok('24 cells (6 stories × 4 categories)', cells().length === 24, cells().length);
  t.ok('no preference on the initial design', cells().every(c => c.classList.contains('empty')));
  t.ok('top row is the top story', d.querySelector('#heat .rh').textContent === '6F');
  t.ok('DQN heatmap is labelled Q value', d.getElementById('heattitle').textContent.includes('Q value'));
  t.ok('DQN heatmap never says probability', !d.getElementById('heattitle').textContent.toLowerCase().includes('probability'));

  click(d, 'next'); click(d, 'next'); click(d, 'next');  // Design Step 3: group 0 is at its minimum section
  const step = fixtures.dqn.steps[2];
  t.ok('fixture sanity: this step has an infeasible group', step.infeasible.length > 0);
  const na = cells().filter(c => c.classList.contains('na')).map(c => Number(c.dataset.group));
  t.ok('infeasible groups are marked unavailable', JSON.stringify(na.sort()) === JSON.stringify(step.infeasible), JSON.stringify(na));
  const chosen = d.querySelectorAll('#heat .cell.chosen');
  t.ok('exactly one chosen cell', chosen.length === 1);
  t.ok('chosen cell is the chosen group', Number(chosen[0].dataset.group) === step.action);
  t.ok('chosen cell shows its raw Q value', chosen[0].textContent === step.preference[step.action].toFixed(3), chosen[0].textContent);
  t.ok('chosen (highest) cell uses the darkest step', chosen[0].classList.contains('p6'));

  runButton(d, 'PPO').click();
  click(d, 'next');
  t.ok('PPO heatmap is labelled action probability', d.getElementById('heattitle').textContent.includes('action probability'));
  const c = d.querySelector('#heat .cell.chosen');
  const s0 = fixtures.ppo.steps[0];
  t.ok('PPO chosen cell shows a percentage', c.textContent === (s0.preference[s0.action] * 100).toFixed(1) + '%', c.textContent);
  return t.result();
};
