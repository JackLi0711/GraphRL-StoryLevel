// 播放控制：起始暫停於初始設計、上/下一步、滑桿、鍵盤、夾在兩端、播放到底自動停止。
const { loadPage, suite, sleep, key, status, cursor, click, fixtures } = require('./lib');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;
  const last = fixtures.dqn.steps.length + 1;  // initial + steps + rolled-back final design

  t.ok('starts on the initial design', status(d).includes('Initial design'));
  t.ok('starts paused', d.getElementById('play').textContent.includes('Play'));
  t.ok('cursor starts at 0', cursor(d) === 0);
  t.ok('slider spans every position', Number(d.getElementById('slider').max) === last, d.getElementById('slider').max);

  click(d, 'prev');
  t.ok('previous at the start stays at 0', cursor(d) === 0);
  click(d, 'next');
  t.ok('next goes to Design Step 1', status(d).includes('Design Step 1 of ' + fixtures.dqn.steps.length));
  key(w, 'ArrowRight');
  t.ok('→ steps forward', cursor(d) === 2);
  key(w, 'ArrowLeft');
  t.ok('← steps back', cursor(d) === 1);

  const slider = d.getElementById('slider');
  slider.value = String(last);
  slider.dispatchEvent(new w.Event('input', { bubbles: true }));
  t.ok('slider jumps to the end', cursor(d) === last);
  click(d, 'next');
  key(w, 'ArrowRight');
  t.ok('next at the end stays at the end', cursor(d) === last);

  // 焦點在滑桿時，方向鍵交給滑桿本身處理，不重複前進
  key(w, 'ArrowLeft', slider);
  t.ok('arrow keys on the focused slider are left to the slider', cursor(d) === last);

  key(w, ' ');
  t.ok('Space starts playback (restarting from the initial design at the end)', d.getElementById('play').textContent.includes('Pause') && cursor(d) === 0);
  key(w, ' ');
  t.ok('Space pauses again', d.getElementById('play').textContent.includes('Play'));

  d.getElementById('speed').value = '4';
  click(d, 'play');
  await sleep(900 / 4 * (last + 2));
  t.ok('playback stops by itself at the last position', cursor(d) === last && d.getElementById('play').textContent.includes('Play'), cursor(d));
  w.close();
  return t.result();
};
