// 切換 Run 會重設到初始設計並暫停；摘要卡片比較兩個 Run。
const { loadPage, suite, status, cursor, click, runButton, fixtures } = require('./lib');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;

  t.ok('DQN selected first', runButton(d, 'DQN').getAttribute('aria-pressed') === 'true');
  click(d, 'next'); click(d, 'next'); click(d, 'play');
  runButton(d, 'PPO').click();
  t.ok('switching Run resets to the initial design', cursor(d) === 0 && status(d).includes('Initial design'));
  t.ok('switching Run pauses', d.getElementById('play').textContent.includes('Play'));
  t.ok('PPO now selected', runButton(d, 'PPO').getAttribute('aria-pressed') === 'true');
  t.ok('Run folder name shown', d.getElementById('runname').textContent === fixtures.ppo.run.name);

  const slider = d.getElementById('slider');
  t.ok('a minimum-section Rollout has no extra final position', Number(slider.max) === fixtures.ppo.steps.length);
  slider.value = slider.max; slider.dispatchEvent(new w.Event('input', { bubbles: true }));
  t.ok('last step says the Rollout is complete', status(d).includes('All groups at minimum section'), status(d));

  const rows = [...d.querySelectorAll('#sumbody tr')];
  t.ok('summary has one row per Run', rows.length === 2);
  const [dqnRow, ppoRow] = rows.map(r => r.textContent);
  for (const [row, doc] of [[dqnRow, fixtures.dqn], [ppoRow, fixtures.ppo]]) {
    t.ok(doc.run.algorithm + ' Saving Ratio', row.includes((doc.end.saving_ratio * 100).toFixed(1) + '%'), row);
    t.ok(doc.run.algorithm + ' Design Step count', row.includes(String(doc.steps.length)));
    t.ok(doc.run.algorithm + ' volumes', row.includes(doc.initial.volume_m3.toFixed(2) + ' → ' + doc.end.final_volume_m3.toFixed(2)), row);
  }
  t.ok('DQN end reason names the failed check', dqnRow.includes('Rejected Step: Story drift ratio'));
  t.ok('PPO end reason is minimum section', ppoRow.includes('All groups at minimum section'));
  t.ok('current Run row highlighted', rows[1].classList.contains('current') && !rows[0].classList.contains('current'));
  w.close();

  // 深層連結：#run=PPO&step=3 直接開在 PPO 的 Design Step 3
  const w3 = loadPage({ url: 'http://localhost/#run=PPO&step=3' });
  const d3 = w3.document;
  t.ok('deep link opens the requested Run', runButton(d3, 'PPO').getAttribute('aria-pressed') === 'true');
  t.ok('deep link opens the requested step', status(d3).includes('Design Step 3 of ' + fixtures.ppo.steps.length), status(d3));
  w3.close();
  return t.result();
};
