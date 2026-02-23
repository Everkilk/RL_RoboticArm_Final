# Random Object Spawning Implementation Summary

## Overview
Modified the Franka-Shadow lift task environment to randomly spawn either a cubic or mustard bottle object for each environment instance, with the unselected object being hidden (invisible and non-interactable).

## Changes Made

### 1. Environment Configuration ([env_cfg.py](env_cfg.py))
- **Changed**: Replaced single `object` configuration with two separate objects:
  - `object_cubic`: DexCube configuration
  - `object_mustard`: YCB Mustard Bottle configuration
- Both objects are now spawned in the scene with different prim paths

### 2. Event System ([mdp/events.py](mdp/events.py))
Added three new event functions:

#### `reset_random_object_selection()`
- Randomly selects which object (0=cubic, 1=mustard) for each environment
- Stores selection in `env.active_object_type` tensor
- Hides non-selected objects by moving them to position (100, 100, -100)
- Called during environment reset with mode='reset'

#### Updated `reset_object_poses()`
- Now resets only the active object for each environment
- Uses `env.active_object_type` to determine which object to reset
- Cubic environments: resets `object_cubic`
- Mustard environments: resets `object_mustard`

#### Updated `clip_object_ranges()`
- Clips positions only for active objects
- Separates cubic and mustard environments for independent clipping

### 3. Observation System ([mdp/observation.py](mdp/observation.py))
Added wrapper functions that automatically select the correct object:

#### `get_active_object_name()`
- Helper function to get active object name for an environment

#### Updated `get_object_position()`
- Returns position of active object for each environment
- Internally splits computation between cubic and mustard environments

#### Updated `get_object_orientation()`
- Returns orientation of active object for each environment
- Internally splits computation between cubic and mustard environments

### 4. Command System ([mdp/command.py](mdp/command.py))
#### Modified `PositionCommand.__init__()`
- Now stores both `object_cubic` and `object_mustard` references
- Maintains backward compatibility with `self.object`

#### Updated `_update_metrics()`
- Uses observation functions (`get_object_position`, `get_object_orientation`) that handle object selection automatically

#### Updated `_resample_command()`
- Uses observation functions to get active object data

#### Updated `_debug_vis_callback()`
- Visualization now shows only active objects per environment
- Creates combined position/quaternion arrays based on `active_object_type`

## How It Works

### Initialization Flow
1. Environment spawns both `object_cubic` and `object_mustard` in all environments
2. During reset, `select_random_object` event randomly assigns each environment to use cubic (0) or mustard (1)
3. Non-selected objects are moved far away (100, 100, -100) to make them invisible and non-interactable
4. `reset_objects` event then positions only the active objects

### During Training
- Each environment independently uses its assigned object
- Observation functions automatically return data for the active object
- Reward and command systems work transparently through observations
- On episode reset, object selection is randomized again

### Distribution Example
With 10 environments:
- Random selection might result in: 4 cubic + 6 mustard, or 7 cubic + 3 mustard, etc.
- Each environment's selection is independent and random

## Files Modified
1. `franka_env/task/lift/env_cfg.py` - Scene configuration with two objects
2. `franka_env/task/lift/mdp/events.py` - Random selection and reset logic
3. `franka_env/task/lift/mdp/observation.py` - Active object observation wrappers
4. `franka_env/task/lift/mdp/command.py` - Command system updates for dual objects

## Backward Compatibility
- All existing reward and termination logic works without modification
- Observation interface remains the same
- Command interface remains the same
- Only internal object selection logic changed

## Testing Recommendations
1. Verify both objects spawn correctly in simulation
2. Check that non-active objects are hidden (far away)
3. Confirm observations return correct object data per environment
4. Verify visualization shows correct active objects
5. Test that training works with mixed object types
