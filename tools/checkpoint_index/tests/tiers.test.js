// 比較區的參數依三層欄位分組（spec C4）。
// 重點：不同演算法並排時，一方沒有的演算法專屬參數不得被當成「相異」。
const { loadPage, suite, rowByName } = require('./lib');

module.exports = async () => {
  const t = suite();
  const w = loadPage();
  const d = w.document;
  const RUNS = w.eval('RUNS');
  const F = w.eval('FIELDS');
  const dqn = RUNS.find(r => r.algorithm === 'DQN');
  const ppo = RUNS.find(r => r.algorithm === 'PPO');
  if (!dqn || !ppo) { t.ok('fixture: need a DQN and a PPO run', false); w.close(); return t.result(); }
  [dqn, ppo].forEach(r => rowByName(d, r.name).querySelector('input[type=checkbox]').click());

  const body = d.getElementById('cmpbody');
  const keysIn = el => el ? [...el.querySelectorAll('tbody tr')].map(tr => tr.firstElementChild.textContent) : [];
  const diffBox = body.querySelector('.diffgroup');
  const same = body.querySelector('details.samegroup');
  const inv = body.querySelector('details.invariantgroup');
  const diffKeys = keysIn(diffBox), sameKeys = keysIn(same), invKeys = keysIn(inv);
  const chosenKeys = [...new Set([dqn, ppo].flatMap(r => Object.keys(r.args)))];
  const inBoth = k => k in dqn.args && k in ppo.args;

  t.ok('every param of the chosen runs appears exactly once across the three groups',
    chosenKeys.every(k => [diffKeys, sameKeys, invKeys].filter(g => g.includes(k)).length === 1)
    && diffKeys.length + sameKeys.length + invKeys.length === chosenKeys.length,
    diffKeys.length + '+' + sameKeys.length + '+' + invKeys.length + ' / ' + chosenKeys.length);

  t.ok('never-varying params are only in the 從不變動 group', invKeys.every(k => F.invariant.includes(k))
    && chosenKeys.filter(k => F.invariant.includes(k)).every(k => invKeys.includes(k)));
  t.ok('從不變動 group is collapsed by default', inv && inv.open === false);
  t.ok('相同參數 group is collapsed by default', same && same.open === false);

  const oneSided = diffKeys.filter(k => !inBoth(k));
  t.ok('a param only one run has is not counted as differing', oneSided.length === 0, oneSided.join(', ') || 'none');
  const shouldDiffer = chosenKeys.filter(k => !F.invariant.includes(k) && inBoth(k)
    && JSON.stringify(dqn.args[k]) !== JSON.stringify(ppo.args[k]));
  t.ok('params both runs have with different values are in 相異',
    shouldDiffer.every(k => diffKeys.includes(k)) && diffKeys.length === shouldDiffer.length,
    diffKeys.join(', '));

  const heading = body.querySelectorAll('h3');
  const diffH3 = [...heading].find(h => h.textContent.startsWith('相異參數'));
  t.ok('相異參數 count matches rows', diffH3 && diffH3.textContent.includes('（' + diffKeys.length + '）'));

  // 小標題依層級：每張表前面的 h4 要與表內參數的層級一致
  const tierOf = k => F.invariant.includes(k) ? 'inv' : F.universal.includes(k) ? '通用'
    : Object.entries(F.algorithm_specific).find(([, ks]) => ks.includes(k))[0];
  const groupsOk = [diffBox, same].every(box => [...box.querySelectorAll('h4.tierh')].every(h4 => {
    const table = h4.nextElementSibling;
    const tiers = new Set(keysIn(table).map(tierOf));
    const label = h4.textContent;
    return tiers.size === 1 && label.includes([...tiers][0] === '通用' ? '通用' : '（' + [...tiers][0] + '）');
  }));
  t.ok('each tier sub-heading matches the tier of the params under it', groupsOk);

  const algoRow = [...same.querySelectorAll('tbody tr'), ...diffBox.querySelectorAll('tbody tr')]
    .find(tr => { const k = tr.firstElementChild.textContent; return !inBoth(k); });
  t.ok('a one-sided algorithm-specific param shows — for the run without it',
    !algoRow || [...algoRow.querySelectorAll('td.diffval')].some(td => td.textContent === '—'));

  // 同一演算法的兩個 Run：專屬參數若值不同，仍應列為相異
  rowByName(d, dqn.name).querySelector('input[type=checkbox]').click();
  rowByName(d, ppo.name).querySelector('input[type=checkbox]').click();
  const ppos = RUNS.filter(r => r.algorithm === 'PPO');
  let pair = null;
  outer: for (let i = 0; i < ppos.length; i++) for (let j = i + 1; j < ppos.length; j++) {
    const k = Object.keys(ppos[i].args).find(k => !F.universal.includes(k) && !F.invariant.includes(k)
      && k in ppos[j].args && JSON.stringify(ppos[i].args[k]) !== JSON.stringify(ppos[j].args[k]));
    if (k) { pair = [ppos[i], ppos[j], k]; break outer; }
  }
  if (pair) {
    pair.slice(0, 2).forEach(r => rowByName(d, r.name).querySelector('input[type=checkbox]').click());
    t.ok('same-algorithm runs: differing algorithm-specific param is in 相異 (' + pair[2] + ')',
      keysIn(d.getElementById('cmpbody').querySelector('.diffgroup')).includes(pair[2]));
  }

  w.close();
  return t.result();
};
