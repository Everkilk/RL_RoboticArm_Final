# Giải thích file `drl/learning/rher.py`

Tài liệu này tổng hợp các thành phần chính của `drl/learning/rher.py` (phiên bản trong workspace).

## Mục đích
- `RHER` là một framework huấn luyện dùng Hindsight Experience Replay cho bài toán nhiều giai đoạn (multi-stage). Lớp `RHER` kế thừa `RLFrameWork` và quản lý luồng: tạo rollout, lưu vào bộ nhớ (memory), huấn luyện agent và đánh giá.

## Các import chính
- `torch`, `numpy`, `tqdm`, `SummaryWriter` — thư viện ML, xử lý tensor và ghi log TensorBoard.
- Từ dự án: `RLFrameWork`, `map_structure`, `put_structure`, `groupby_structure`, `nearest_node_value`, `MeanMetrics`, `LOGGER`, `format_time`, `format_tabulate` — các tiện ích và kiểu cơ sở.

## Kiến trúc lớp `RHER`
- `__init__(envs, agent, memory, compute_metrics=None)`
  - Kiểm tra interface của `agent` và `memory` (ví dụ `memory.reward_func`, `memory.num_stages`, `memory.horizon`).
  - `compute_metrics` mặc định: kết hợp `goal_achieved` và tỉ lệ reward/horizon để tạo `eval_value` tổng hợp.

### Exploration / stage handling
- `_get_stages(observations, mt_goals)`
  - Gọi `memory.reward_func` cho từng `stage_id` để nhận `goal_achieved` tại mọi stage.
  - Tính `stages` hiện tại cho mỗi phần tử trong batch bằng cách xét stage lớn nhất đã đạt.

- `_select_action(observations, mt_goals, stages, *, r_mix=0.5)`
  - `r_mix` (0..1) là xác suất để "trộn" goal: với xác suất `r_mix` mẫu sẽ được chuyển từ stage hiện tại sang stage kế tiếp trước khi chọn action.
  - Mục đích: hướng dẫn exploration theo mục tiêu (goal-directed exploration) bằng cách đôi khi cung cấp goal của stage tiếp theo.
  - Gọi `agent({'observation':..., 'goal':..., 'task_id': stages}, deterministic=False)` để lấy action non-deterministic.

- `select_actions(..., deterministic=False)`
  - Nếu không deterministic: chuẩn hoá dữ liệu bằng `agent.format_data`, gọi `_get_stages` và `_select_action`, trả về `(actions_numpy, goal_achieveds_bool)`.
  - Nếu deterministic: dùng goal cuối cùng và stage cuối, gọi policy deterministic.

### Training
- `train(num_updates, batch_size, future_p=0.8, n_steps=1, step_decay=0.7, discounted_factor=0.99, clip_return=None)`
  - Nếu dùng prioritized replay thì cập nhật priorities trước.
  - Lặp `num_updates` lần: sample từ `memory.sample(...)` (ở đây HER sampling được xử lý trong memory), gọi `agent.update(...)` để cập nhật mạng.
  - Trả về `MeanMetrics` chứa các metric trung bình.

### Evaluate
- `evaluate(num_episodes=10)`
  - Chạy envs ở chế độ `eval`, chọn action deterministic, thu thông tin episode (reward, horizon, infos), sử dụng `compute_metrics` để tính `eval_value`.

### Run (luồng chính)
- `run(...)`
  - Tạo thư mục experiment (`make_exp_dir`), có thể `resume` từ checkpoint.
  - Với mỗi epoch: lặp cycles, mỗi cycle gọi `generate_rollouts`, `store_rollouts`, sau đó `train` nhiều lần.
  - Sau epoch chạy `evaluate` và lưu policy/ckpt nếu cần.

### Rollout / Memory
- `generate_rollouts(r_mix='auto')`
  - Tạo rollout theo `memory.horizon`. Mỗi step: gọi `select_actions` (non-deterministic), thực hiện `envs.step`, thu observations, action, và `goal_achieved`.
  - Trả về batch rollouts (cấu trúc đã swap axis) và `info` (mean eps_reward, eps_horizon, ...).

- `store_rollouts(rollouts)`
  - Chuẩn hoá các chiều mảng (`meta`, `achieved_goal`, `desired_goal`) trước khi gọi `memory.store(rollouts)`.

## Về `r_mix` và entropy
  - `r_mix`: exploration theo goal (thay đổi input mục tiêu để agent dẫn hướng tới stage kế tiếp).
  - Entropy: exploration ở action-space (policy thử nhiều action khác nhau cho cùng goal).

## Ví dụ cụ thể về `r_mix` và entropy trong huấn luyện cánh tay robot

### Bối cảnh mẫu
- Task: pick-and-place chia thành 4 stage: `reach` → `grasp` → `lift` → `place`.
- `memory.num_stages = 4`, mỗi rollout có `memory.horizon` bước.
- Agent có thể là SAC (policy Gaussian, có hệ số entropy `alpha`) hoặc DDPG/TD3 (thêm noise với std).

### Ý tưởng chính (tóm tắt)
- `r_mix`: xác suất tại mỗi bước để thay goal hiện tại bằng goal của stage tiếp theo trước khi policy chọn action (goal-guided exploration).
- Entropy / noise: điều chỉnh mức độ ngẫu nhiên của policy khi chọn action cho cùng một (state, goal) (action-space exploration).

### Cấu hình mẫu và hành vi mong đợi
- Cấu hình A — Thận trọng:
  - `r_mix = 0.0`, entropy thấp / noise nhỏ (SAC `alpha ≈ 0.01`, DDPG noise std ≈ 0.02).
  - Agent học chắc từng stage, ít thử nghiệm goal kế tiếp, ổn định nhưng có thể chậm qua stage.

- Cấu hình B — Cân bằng:
  - `r_mix = 0.5`, entropy moderate (SAC `alpha ≈ 0.05 - 0.2`, DDPG std ≈ 0.05 - 0.15).
  - Một phần rollout hướng tới mục tiêu tiếp theo, policy vẫn thử nhiều hành động — có lợi cho chuyển tiếp giữa stage.

- Cấu hình C — Hướng mạnh mục tiêu tiếp theo:
  - `r_mix = 1.0`, entropy thấp (nếu entropy cao đồng thời có thể gây bất ổn).
  - Agent liên tục nhận goal của stage tiếp theo, đẩy nhanh học chuyển giai đoạn nhưng có rủi ro nếu kỹ năng cơ bản yếu.

- Thử nghiệm khám phá rộng:
  - `r_mix = 0.8` kết hợp entropy cao (SAC `alpha` lớn hoặc noise std cao) → khám phá rất rộng, khó hội tụ nhưng có thể tìm chuỗi hành động phức tạp.

### Ví dụ số cho pick-and-place
- Setup: `batch_size=256`, `horizon=50`, `num_stages=4`.
- Grid thử: `r_mix` ∈ {0.0, 0.25, 0.5, 0.75, 1.0} × SAC `alpha` ∈ {0.01, 0.05, 0.2} (hoặc DDPG std ∈ {0.02, 0.08, 0.2}).
- Theo dõi: success rate per stage, time-to-first-success, average reward, variance. Chạy nhiều seed để kiểm nghiệm.

### Cách cấu hình trong code
- Truyền `r_mix` khi gọi `run()`:

```python
# ví dụ
rher.run(..., r_mix=0.5, num_updates=50, batch_size=256)
```

- Cấu hình entropy cho SAC (ví dụ):

```python
# nếu agent là SAC
sac_agent = SACAgent(..., entropy_coef=0.05)
```

- Cấu hình noise cho DDPG:

```python
ddpg_agent = DDPGAgent(..., action_noise_std=0.08)
```

### Lưu ý tuning & thử nghiệm đề xuất
- Bắt đầu với `r_mix=0.25-0.5` và entropy moderate (SAC alpha auto hoặc ~0.05). Quan sát: nếu agent không tiến qua stage 1→2 thì tăng `r_mix`; nếu agent thất bại nhiều ở stage hiện tại thì giảm `r_mix` hoặc giảm entropy.
- Thử một biến tại một thời điểm (chỉ đổi `r_mix` hoặc chỉ đổi entropy) để thấy tác động rõ ràng.
- Thử nghiệm đề xuất (ngắn):
  1. Baseline: `r_mix=0.0`, SAC `alpha=0.05`.
  2. Thêm `r_mix=0.5`, giữ `alpha` cố định — so sánh success per stage.
  3. Giữ `r_mix=0.5`, thay đổi `alpha` lên/xuống.

### Cảnh báo
- `generate_rollouts` trong repo có default `r_mix='auto'` nhưng `_select_action` mong `r_mix` là số trong `[0,1]`. Hãy đảm bảo khi gọi `generate_rollouts()` truyền giá trị số thực hoặc sửa `generate_rollouts` để xử lý `'auto'` hợp lý.

---

Phần này đã bổ sung các ví dụ và hướng dẫn tuning. Nếu muốn, tôi sẽ:
- Thêm một script mẫu để chạy grid-search `r_mix × alpha` và lưu metrics, hoặc
- Sửa `generate_rollouts` để `r_mix='auto'` mặc định về `0.5` và thêm docstring giải thích.

## Lưu ý thực dụng
- `memory.reward_func` phải hỗ trợ `stage_id` (multi-stage reward).
- `generate_rollouts` mặc định `r_mix='auto'` nhưng `_select_action` cần `r_mix` là số trong `[0,1]`. Nếu gọi `generate_rollouts()` mà không truyền `r_mix` thì `'auto'` có thể gây lỗi. Nên sửa để xử lý `'auto'` hoặc luôn truyền `r_mix` từ `run()`.
- `agent` phải cung cấp `format_data`, `device`, `update`, và chấp nhận input `{'observation','goal','task_id'}`.
- `memory` phải implement `store`, `sample`, `reward_func`, `num_stages`, `horizon`, và nếu dùng prioritized replay thì `use_priority` + `update_priorities()`.

## Giải thích 'rollout' (đơn giản và cụ thể)

- Rollout = chuỗi các tương tác (transitions) thu được khi chạy policy trong môi trường: observations, actions, rewards, next-observations, và thông tin liên quan (done, infos, achieved/desire goals...).
- Ở repo này, `generate_rollouts` tạo rollout có độ dài `memory.horizon` (còn `envs.num_envs` là số môi trường song song).

### Shapes & lý do cắt dữ liệu
- Với `num_envs = N`, `horizon = H`, sau `generate_rollouts` ta có (ví dụ):
  - `rollouts['observation']`: shape `(N, H+1, ...)` — bao gồm observation ban đầu và H lần next-observation.
  - `rollouts['action']`: shape `(N, H, ...)` — hành động tại mỗi step.
  - `rollouts['achieved_goal']`: shape `(N, H+1, ...)`.
  - `rollouts['desired_goal']`: thường thu được `(N, H+1, ...)` nhưng khi store sẽ cắt thành `(N, H, ...)` tương ứng với actions.

- Tại sao `store_rollouts` cắt/move các mảng:
  - Khi lưu transitions cần khớp `(state_t, action_t, reward_t, next_state_{t+1})`.
  - Do đó `achieved_goal` được cắt `[:, 1:]` (bỏ achieved_goal lúc time=0) để khớp với `action_t` → chính là achieved ở next state.
  - `desired_goal` được cắt `[:, :-1]` (bỏ goal cuối) để mỗi `action_t` có `desired_goal` tương ứng ở thời điểm t.
  - Nếu có `meta`, cũng cắt bỏ phần đầu để khớp tương tự.

### Vai trò của rollout trong HER / RHER
- Rollout là nguồn dữ liệu thô; HER thay nhãn (relabel) goals trong memory khi sample (đổi desired goal thành achieved goal ở một thời điểm sau trong cùng rollout) để học từ trải nghiệm "hindsight".
- Ở RHER, `r_mix` có tác động ngay khi tạo rollout: một số step sẽ dùng goal của stage tiếp theo khi chọn action, vì vậy rollout phản ánh cả việc trộn goal và exploration hướng mục tiêu.

### Ví dụ ngắn (pick-and-place)
- Nếu `N=8`, `H=50`: `generate_rollouts` thu `8×51` observations, `8×50` actions, `8×51` achieved_goal. Sau `store_rollouts` lưu vào memory các mảng dạng `8×50` cho transitions `(state_t, action_t, next_state, desired_goal_t, achieved_goal_next, goal_achieved_flag)`.
