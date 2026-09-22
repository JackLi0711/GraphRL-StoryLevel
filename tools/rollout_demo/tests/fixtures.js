// Fixture Rollout documents (schema_version 1) on the Testing Geometry topology.
// DQN: starts at the largest sections and ends with a Rejected Step (drift ratio).
// PPO: starts one step above minimum and ends with every group at its minimum section.
const CATEGORIES = ['xdir_beam', 'zdir_beam', 'outer_column', 'inner_column'];
const KIND = { xdir_beam: 'beam', zdir_beam: 'beam', outer_column: 'column', inner_column: 'column' };
const BAYS = 4, STORIES = 6, SPAN_X = 6000, SPAN_Z = 8000, H1 = 4200, H = 3200;
const GROUPS = CATEGORIES.length * STORIES;
const groupOf = (category, story) => CATEGORIES.indexOf(category) * STORIES + story - 1;

function topology() {
  const nodes = [], id = {};
  const levelY = l => (l === 0 ? 0 : H1 + (l - 1) * H);
  for (let l = 0; l <= STORIES; l++)
    for (let j = 0; j <= BAYS; j++)
      for (let i = 0; i <= BAYS; i++) { id[[i, j, l]] = nodes.length; nodes.push([i * SPAN_X, levelY(l), j * SPAN_Z]); }
  const members = [];
  for (let s = 1; s <= STORIES; s++) {
    for (let j = 0; j <= BAYS; j++) for (let i = 0; i <= BAYS; i++) {
      const outer = i === 0 || j === 0 || i === BAYS || j === BAYS;
      members.push([id[[i, j, s - 1]], id[[i, j, s]], groupOf(outer ? 'outer_column' : 'inner_column', s)]);
    }
    for (let j = 0; j <= BAYS; j++) for (let i = 0; i < BAYS; i++) members.push([id[[i, j, s]], id[[i + 1, j, s]], groupOf('xdir_beam', s)]);
    for (let j = 0; j < BAYS; j++) for (let i = 0; i <= BAYS; i++) members.push([id[[i, j, s]], id[[i, j + 1, s]], groupOf('zdir_beam', s)]);
  }
  return { nodes, members };
}

const groups = [];
CATEGORIES.forEach(c => { for (let s = 1; s <= STORIES; s++) groups.push({ story: s, category: c, kind: KIND[c] }); });
const catalog = {
  beam: Array.from({ length: 15 }, (_, i) => 'B' + (300 + i * 20) + 'x200'),
  column: Array.from({ length: 15 }, (_, i) => 'C' + (350 + i * 20) + 'x20'),
};
const volume = sections => 10 + sections.reduce((a, b) => a + b, 0) * 0.3;

function rollout({ algorithm, name, initial, actions, rejection, preference }) {
  const initialVolume = volume(initial);
  let sections = initial.slice();
  const steps = actions.map((action, k) => {
    const infeasible = sections.map((s, g) => (s === 0 ? g : -1)).filter(g => g >= 0);
    const before = sections[action];
    const after = sections.slice();
    after[action] -= 1;
    const passed = !(rejection && k === actions.length - 1);
    const vol = volume(after);
    const rec = {
      index: k,
      preference: sections.map((s, g) => (s === 0 ? null : preference(g, action, infeasible.length))),
      infeasible,
      action,
      section_before: before,
      section_after: before - 1,
      sections_after: after,
      volume_m3_after: vol,
      saving_ratio_after: passed ? (initialVolume - vol) / initialVolume : null,
      passed,
      rejection: passed ? null : rejection,
    };
    if (passed) sections = after;
    return rec;
  });
  const finalVolume = volume(sections);
  return {
    schema_version: 1,
    run: { name, path: 'Results/Fixture/' + name, algorithm, model_file: 'model_HighestScore.pt' },
    geometry: { x_span_num: BAYS, z_span_num: BAYS, story_num: STORIES, x_span_lens: Array(BAYS).fill(SPAN_X), z_span_lens: Array(BAYS).fill(SPAN_Z), story_height: H },
    topology: topology(),
    groups,
    section_catalog: catalog,
    preference_kind: algorithm === 'DQN' ? 'q_value' : 'action_probability',
    initial: { sections: initial, volume_m3: initialVolume },
    steps,
    end: {
      reason: rejection ? 'rejected' : 'minimum_section',
      final_sections: sections,
      final_volume_m3: finalVolume,
      saving_ratio: (initialVolume - finalVolume) / initialVolume,
    },
  };
}

const DRIFT = { check: 'drift_ratio', load_case: 'EQX+', value: 0.0061345, limit: 0.005, comparator: '>' };

// group 0 (1F X beam) goes to its minimum at step 1, so it is infeasible from step 2 on
const dqnInitial = Array(GROUPS).fill(14);
dqnInitial[0] = 2;
const dqn = rollout({
  algorithm: 'DQN', name: '2026_01_05__11_59_47__DouDQN_Fixture', initial: dqnInitial,
  actions: [0, 0, 5, 11, 17], rejection: DRIFT,
  preference: (g, action) => (g === action ? 3.5 : 1 + g * 0.05),
});
const ppo = rollout({
  algorithm: 'PPO', name: '2026_03_10__13_19_18__PPO_Fixture', initial: Array(GROUPS).fill(1),
  actions: Array.from({ length: GROUPS }, (_, g) => GROUPS - 1 - g), rejection: null,
  preference: (g, action, nInfeasible) => (g === action ? 0.6 : 0.4 / Math.max(1, GROUPS - nInfeasible - 1)),
});

module.exports = { dqn, ppo, GROUPS, DRIFT };
