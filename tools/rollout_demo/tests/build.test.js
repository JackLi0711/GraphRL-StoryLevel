// 建置結果：單一自足的 HTML、DQN 排在 PPO 前面、名稱經過跳脫。
const { build, loadPage, suite, fixtures } = require('./lib');

module.exports = async () => {
  const t = suite();
  const html = build([fixtures.ppo, fixtures.dqn]);
  t.ok('no external src/href URLs', !/(src|href)\s*=\s*["']?(https?:)?\/\//i.test(html));
  t.ok('no fetch / XHR', !/\bfetch\s*\(|XMLHttpRequest/.test(html));
  const w = loadPage({ html });
  const runs = [...w.document.querySelectorAll('#runs button')].map(b => b.textContent);
  t.ok('Runs ordered DQN then PPO', runs.join(',') === 'DQN,PPO', runs.join(','));

  // Run 名稱含 HTML / </script> 時不得被解讀
  const evil = JSON.parse(JSON.stringify(fixtures.dqn));
  evil.run.name = '<b class="injected">x</b></script><i class="injected"></i>';
  const w2 = loadPage({ html: build([evil, fixtures.ppo]) });
  const d2 = w2.document;
  t.ok('page still renders with a hostile Run name', d2.getElementById('status').textContent.includes('Initial design'));
  t.ok('no injected element anywhere', d2.querySelectorAll('.injected').length === 0);
  t.ok('Run name shown as text', d2.getElementById('runname').textContent === evil.run.name);
  t.ok('summary shows the Run name as text', d2.getElementById('sumbody').textContent.includes(evil.run.name));
  return t.result();
};
