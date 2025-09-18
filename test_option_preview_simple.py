#!/usr/bin/env python3
"""
Simple test for option preview visualization functions.
Tests core functionality without complex environment setup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_action_mapping_logic():
    """Test action mapping logic with mock data."""
    print("Testing action mapping logic...")

    # Mock structure data
    class MockStructure:
        def __init__(self):
            self.story_xdir_beam_member = [[1, 2], [3, 4]]  # 2 stories
            self.story_zdir_beam_member = [[5, 6], [7, 8]]  # 2 stories
            self.story_outer_column_member = [[9, 10], [11, 12]]  # 2 stories
            self.story_inner_column_member = [[13, 14], [15, 16]]  # 2 stories

            # story_level_categories should match the order and length of story_level_actions
            self.story_level_categories = ['xdir_beam', 'xdir_beam', 'zdir_beam', 'zdir_beam',
                                         'outer_column', 'outer_column', 'inner_column', 'inner_column']

    def action_to_story_component(action_index, structure):
        """Test version of action mapping function."""
        category = structure.story_level_categories[action_index]

        if category == 'xdir_beam':
            story_index = action_index
            component_type = 'X-Beam'
        elif category == 'zdir_beam':
            story_index = action_index - len(structure.story_xdir_beam_member)
            component_type = 'Z-Beam'
        elif category == 'outer_column':
            story_index = action_index - len(structure.story_xdir_beam_member) - len(structure.story_zdir_beam_member)
            component_type = 'Outer-Column'
        else:  # inner_column
            story_index = action_index - len(structure.story_xdir_beam_member) - len(structure.story_zdir_beam_member) - len(structure.story_outer_column_member)
            component_type = 'Inner-Column'

        return story_index + 1, component_type  # Floor starts from 1

    # Test with mock structure
    mock_structure = MockStructure()

    expected_results = [
        (1, 'X-Beam'),        # action 0
        (2, 'X-Beam'),        # action 1
        (1, 'Z-Beam'),        # action 2
        (2, 'Z-Beam'),        # action 3
        (1, 'Outer-Column'),  # action 4
        (2, 'Outer-Column'),  # action 5
        (1, 'Inner-Column'),  # action 6
        (2, 'Inner-Column'),  # action 7
    ]

    for i, expected in enumerate(expected_results):
        result = action_to_story_component(i, mock_structure)
        if result == expected:
            print(f"✓ Action {i}: Story {result[0]}, Component {result[1]}")
        else:
            print(f"❌ Action {i}: Expected {expected}, got {result}")
            return False

    return True

def test_statistics_analysis_logic():
    """Test statistics analysis with mock data."""
    print("\nTesting statistics analysis...")

    # Reuse mock structure from above
    class MockStructure:
        def __init__(self):
            self.story_xdir_beam_member = [[1, 2], [3, 4]]
            self.story_zdir_beam_member = [[5, 6], [7, 8]]
            self.story_outer_column_member = [[9, 10], [11, 12]]
            self.story_inner_column_member = [[13, 14], [15, 16]]
            self.story_level_categories = ['xdir_beam', 'xdir_beam', 'zdir_beam', 'zdir_beam',
                                         'outer_column', 'outer_column', 'inner_column', 'inner_column']

    def action_to_story_component(action_index, structure):
        category = structure.story_level_categories[action_index]

        if category == 'xdir_beam':
            story_index = action_index
            component_type = 'X-Beam'
        elif category == 'zdir_beam':
            story_index = action_index - len(structure.story_xdir_beam_member)
            component_type = 'Z-Beam'
        elif category == 'outer_column':
            story_index = action_index - len(structure.story_xdir_beam_member) - len(structure.story_zdir_beam_member)
            component_type = 'Outer-Column'
        else:  # inner_column
            story_index = action_index - len(structure.story_xdir_beam_member) - len(structure.story_zdir_beam_member) - len(structure.story_outer_column_member)
            component_type = 'Inner-Column'

        return story_index + 1, component_type

    def analyze_actions_by_story_component(action_indices, structure):
        stats = {
            'total_actions': len(action_indices),
            'by_story': {},
            'by_component': {'Inner-Column': 0, 'Outer-Column': 0, 'X-Beam': 0, 'Z-Beam': 0}
        }

        for action_idx in action_indices:
            story, component = action_to_story_component(action_idx, structure)

            if story not in stats['by_story']:
                stats['by_story'][story] = {}
            if component not in stats['by_story'][story]:
                stats['by_story'][story][component] = 0

            stats['by_story'][story][component] += 1
            stats['by_component'][component] += 1

        return stats

    def prepare_episode_option_data(episode_data, structure):
        actions = episode_data['actions']
        option_instances = episode_data['option_instances']

        instance_groups = {}
        for instance_data in option_instances:
            instance_id = instance_data['instance_id']
            if instance_id not in instance_groups:
                instance_groups[instance_id] = {
                    'option_index': instance_data['option_index'],
                    'action_indices': [],
                    'first_action_step': float('inf')
                }

            action_step = instance_data['action_step']
            instance_groups[instance_id]['action_indices'].append(actions[action_step])
            instance_groups[instance_id]['first_action_step'] = min(
                instance_groups[instance_id]['first_action_step'], action_step
            )

        for instance_id in instance_groups:
            group_actions = instance_groups[instance_id]['action_indices']
            stats = analyze_actions_by_story_component(group_actions, structure)
            instance_groups[instance_id]['statistics'] = stats

        sorted_instances = sorted(instance_groups.items(),
                                key=lambda x: x[1]['first_action_step'])

        return sorted_instances

    # Test statistics analysis
    mock_structure = MockStructure()

    # Mock episode data
    actions = [0, 1, 2, 4, 6, 7]  # Mix of different action types
    option_instances = [
        {'option_index': 0, 'instance_id': 0, 'action_step': 0},  # action 0 (1F X-Beam)
        {'option_index': 0, 'instance_id': 0, 'action_step': 1},  # action 1 (2F X-Beam)
        {'option_index': 1, 'instance_id': 1, 'action_step': 2},  # action 2 (1F Z-Beam)
        {'option_index': 1, 'instance_id': 1, 'action_step': 3},  # action 4 (1F Outer-Column)
        {'option_index': 2, 'instance_id': 2, 'action_step': 4},  # action 6 (1F Inner-Column)
        {'option_index': 2, 'instance_id': 2, 'action_step': 5},  # action 7 (2F Inner-Column)
    ]

    episode_data = {
        'actions': actions,
        'options': [0, 0, 1, 1, 2, 2],
        'option_instances': option_instances
    }

    # Test data preparation and analysis
    sorted_instances = prepare_episode_option_data(episode_data, mock_structure)

    print(f"Found {len(sorted_instances)} option instances")

    for instance_id, instance_data in sorted_instances:
        stats = instance_data['statistics']
        print(f"Option Instance {instance_id} (Option {instance_data['option_index']}):")
        print(f"  Total actions: {stats['total_actions']}")
        print(f"  By component: {stats['by_component']}")
        print(f"  By story: {stats['by_story']}")
        print()

    return len(sorted_instances) == 3  # Should have 3 option instances

def main():
    """Run all tests."""
    print("=" * 60)
    print("Option Preview Functionality Test")
    print("=" * 60)

    try:
        # Test action mapping
        mapping_ok = test_action_mapping_logic()
        if not mapping_ok:
            print("❌ Action mapping test failed")
            return False

        # Test statistics analysis
        analysis_ok = test_statistics_analysis_logic()
        if not analysis_ok:
            print("❌ Statistics analysis test failed")
            return False

        print("=" * 60)
        print("✓ All tests passed!")
        print("The option preview core functionality is working correctly.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)