// 測試共用工具：用 fixture Rollout 文件跑 build_demo.py，再在 jsdom 中載入產生的頁面。
//
// 需要能執行 build_demo.py 的 Python（只用標準函式庫）；預設 `python`，可用環境變數 PYTHON 指定。
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { JSDOM } = require('jsdom');
const fixtures = require('./fixtures');

const BUILDER = path.resolve(__dirname, '..', 'build_demo.py');

// docs: Rollout 文件陣列；回傳 build_demo.py 產生的 HTML 字串
function build(docs) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rollout-demo-'));
  docs.forEach((d, i) => fs.writeFileSync(path.join(dir, 'r' + i + '.json'), JSON.stringify(d)));
  const out = path.join(dir, 'out.html');
  const py = process.env.PYTHON || 'python';
  const r = spawnSync(py, [BUILDER, '--data_dir', dir, '--out', out], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error('build_demo.py failed (' + py + '): ' + (r.stderr || r.error));
  const html = fs.readFileSync(out, 'utf8');
  fs.rmSync(dir, { recursive: true, force: true });
  return html;
}

let cached = null;
function defaultHtml() {
  if (!cached) cached = build([fixtures.ppo, fixtures.dqn]);  // builder must order DQN first
  return cached;
}

function loadPage({ html, url } = {}) {
  const dom = new JSDOM(html || defaultHtml(), { runScripts: 'dangerously', url: url || 'http://localhost/', pretendToBeVisual: true });
  return dom.window;
}

function suite() {
  let fail = 0, pass = 0;
  const ok = (name, cond, extra) => {
    if (cond) pass++; else fail++;
    console.log((cond ? 'ok   ' : 'FAIL ') + name + (extra !== undefined ? ' -> ' + extra : ''));
  };
  return { ok, result: () => ({ pass, fail }) };
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
const key = (w, k, target) => (target || w.document).dispatchEvent(new w.KeyboardEvent('keydown', { key: k, code: k === ' ' ? 'Space' : k, bubbles: true }));
const status = d => d.getElementById('status').textContent;
const cursor = d => Number(d.getElementById('pos').textContent.split('/')[0]);
const click = (d, id) => d.getElementById(id).click();
const runButton = (d, alg) => [...d.querySelectorAll('#runs button')].find(b => b.textContent === alg);

module.exports = { build, loadPage, suite, sleep, key, status, cursor, click, runButton, fixtures };
