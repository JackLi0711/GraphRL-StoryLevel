// 測試共用工具：在 jsdom 中載入「實際產生出來的」index.html 並執行其中的 script。
//
// 前置條件：先執行 python tools/checkpoint_index/build_index.py 產生 index.html。
// 測試讀取的是本機 Results/ 的真實資料，因此期望值一律由資料推算，不寫死數量。
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const HTML_PATH = path.resolve(__dirname, '..', 'index.html');

function readHtml() {
  if (!fs.existsSync(HTML_PATH)) {
    throw new Error('找不到 ' + HTML_PATH + '，請先執行 python tools/checkpoint_index/build_index.py');
  }
  return fs.readFileSync(HTML_PATH, 'utf8');
}

// localStorage key 以頁面為準，不在測試裡重複寫死
const LS_KEY = (() => {
  const m = readHtml().match(/const LS_KEY = "([^"]+)"/);
  if (!m) throw new Error('index.html 中找不到 LS_KEY');
  return m[1];
})();

// 以測試用的標註取代頁面內嵌的 annotations.json，
// 讓測試結果不受使用者真實的 annotations.json 影響，也永遠不會寫入它。
function withAnnotations(html, annotations) {
  const re = /^const ANN_EMBEDDED = .*;\r?$/m;
  if (!re.test(html)) throw new Error('index.html 中找不到 ANN_EMBEDDED 這一行');
  const json = JSON.stringify(annotations).replace(/<\//g, '<\\/');
  return html.replace(re, () => 'const ANN_EMBEDDED = ' + json + ';');
}

// transformHtml: 在載入前改寫 HTML（例如替內嵌資料注入真實資料中沒有的情況）
function loadPage({ annotations = { version: 2, annotations: {} }, localStorage = {}, beforeParse, transformHtml } = {}) {
  let html = withAnnotations(readHtml(), annotations);
  if (transformHtml) html = transformHtml(html);
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    url: 'http://localhost/',
    beforeParse(w) {
      for (const [k, v] of Object.entries(localStorage)) w.localStorage.setItem(k, v);
      // jsdom 沒有實作 <dialog> 的 showModal / close
      w.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
      w.HTMLDialogElement.prototype.close = function (v) {
        if (v !== undefined) this.returnValue = v;
        this.open = false;
        this.dispatchEvent(new w.Event('close'));
      };
      if (beforeParse) beforeParse(w);
    },
  });
  return dom.window;
}

function suite() {
  let fail = 0;
  let pass = 0;
  const ok = (name, cond, extra) => {
    if (cond) pass++; else fail++;
    console.log((cond ? 'ok   ' : 'FAIL ') + name + (extra !== undefined ? ' -> ' + extra : ''));
  };
  return { ok, result: () => ({ pass, fail }) };
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// 表格輔助
const rows = d => [...d.querySelectorAll('#body tr')];
const rowByName = (d, prefix) => rows(d).find(tr => tr.querySelector('td.name').textContent.startsWith(prefix));

module.exports = { loadPage, suite, sleep, rows, rowByName, LS_KEY };
