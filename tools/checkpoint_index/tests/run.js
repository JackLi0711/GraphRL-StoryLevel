// checkpoint index 檢視器測試
//
// 執行方式（在 repo 根目錄）：
//   python tools/checkpoint_index/build_index.py      # 先產生 index.html
//   cd tools/checkpoint_index/tests && npm install    # 第一次：安裝 jsdom
//   npm test
//   python -m unittest discover -s tools/checkpoint_index/tests -p "test_*.py"   # build_index.py 的測試（在 repo 根目錄執行）
//
// 測試讀的是本機 Results/ 產生的真實資料（Results/ 不在版控中），
// 並以記憶體中的 fixture 取代標註，不會讀寫 annotations.json。
const SUITES = ['data', 'date', 'sort', 'annotations', 'legacy-annotations', 'comment', 'compare', 'tiers', 'warnings'];

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
