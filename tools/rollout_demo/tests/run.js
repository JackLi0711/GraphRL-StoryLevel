// Rollout demo 頁面測試（seam 2）
//
// 執行方式：
//   cd tools/rollout_demo/tests && npm install    # 第一次：安裝 jsdom
//   npm test                                       # 需要 python（或以 PYTHON=... 指定）來執行 build_demo.py
//
// 頁面一律由 fixtures.js 的 Rollout 文件建出，不讀 data/ 裡的真實匯出。
const SUITES = ['build', 'playback', 'frame', 'heatmap', 'rejected', 'runs'];

(async () => {
  let pass = 0, fail = 0;
  for (const name of SUITES) {
    console.log('\n# ' + name);
    try {
      const r = await require('./' + name + '.test.js')();
      pass += r.pass; fail += r.fail;
    } catch (e) {
      fail++;
      console.log('FAIL suite crashed: ' + (e && e.stack || e));
    }
  }
  console.log('\n' + (fail ? fail + ' FAILED, ' : '') + pass + ' passed');
  process.exitCode = fail ? 1 : 0;
})();
