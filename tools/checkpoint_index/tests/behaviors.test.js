// 比較區的 testing_behaviors.png 區塊：位置、相對路徑、缺圖與連結。
const { loadPage, suite, rowByName } = require('./lib');
const fs = require('fs');
const path = require('path');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;
  const RUNS = w.eval('RUNS');
  const withPng = RUNS.find(r => r.has_behaviors_png);
  const without = RUNS.find(r => !r.has_behaviors_png);
  const chosen = [withPng, without].filter(Boolean);
  chosen.forEach(r => rowByName(d, r.name).querySelector('input[type=checkbox]').click());
  if (chosen.length < 2) {
    // 全部 Run 都有圖時，改選兩個有圖的，缺圖的情況以注入方式測
    RUNS.slice(0, 2).forEach(r => rowByName(d, r.name).querySelector('input[type=checkbox]').click());
  }

  const body = d.getElementById('cmpbody');
  const h3 = [...body.querySelectorAll('h3')].map(h => h.textContent);
  t.ok('testing_behaviors.png section exists', h3.includes('testing_behaviors.png'), h3.join(' | '));
  t.ok('section sits between 指標 and 最終設計',
    h3.indexOf('指標') < h3.indexOf('testing_behaviors.png')
    && h3.indexOf('testing_behaviors.png') < h3.findIndex(x => x.startsWith('最終設計')), h3.join(' | '));

  const cards = [...body.querySelectorAll('.behaviors .bcard')];
  t.ok('one card per chosen run', cards.length === body.querySelectorAll('.gcard').length, cards.length);

  const cardOf = r => cards.find(c => c.querySelector('h4').textContent.startsWith(r.name.slice(0, 40)));
  if (withPng) {
    const img = cardOf(withPng).querySelector('img');
    const expected = '../../' + withPng.id + '/testing_behaviors.png';
    t.ok('img src is a relative path to the run folder', img && img.getAttribute('src') === encodeURI(expected), img && img.getAttribute('src'));
    t.ok('the referenced file actually exists on disk',
      fs.existsSync(path.resolve(__dirname, '..', decodeURI(img.getAttribute('src')))));
    t.ok('image is lazy-loaded', img.getAttribute('loading') === 'lazy');
    t.ok('image links to the full-size file in a new tab', (() => {
      const a = img.parentElement;
      return a.tagName === 'A' && a.getAttribute('href') === img.getAttribute('src') && a.target === '_blank';
    })());
    t.ok('page does not inline the image data', !img.getAttribute('src').startsWith('data:'));
  }
  if (without) {
    t.ok('run without the png shows a message instead of a broken image',
      !cardOf(without).querySelector('img') && cardOf(without).textContent.includes('沒有 testing_behaviors.png'));
  }

  // 圖檔載入失敗時（例如頁面被移走）要顯示可診斷的訊息，而不是破圖
  if (withPng) {
    const img = cardOf(withPng).querySelector('img');
    img.onerror();
    const card = cardOf(withPng);
    t.ok('load failure replaces the image with the path it tried', !card.querySelector('img')
      && card.textContent.includes(withPng.id + '/testing_behaviors.png'), card.textContent.slice(0, 60));
  }

  // 名稱跳脫
  w.eval('RUNS.filter(r => sel.has(r.id)).forEach(r => { r.name = "<b class=injected>x</b>" + r.name; }); compare();');
  t.ok('run name in the card title is escaped', d.querySelectorAll('#cmpbody .behaviors .injected').length === 0);

  w.close();
  return t.result();
};
