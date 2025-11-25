import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# 動作定義
ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_PICKUP = 5
ACTION_DROPOFF = 6

ACTION_MEANING = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "pickup",
    6: "dropoff",
}


class RoboTaxiEnv:
    """
    Multi-Agent RoboTaxi Environment (30x30 grid)

    - 3 taxis
    - 10 passengers
    - Node traffic lights (different cycle lengths)
    - Node & edge random events (blocked)
    """

    def __init__(
        self,
        grid_size: int = 10,
        n_taxis: int = 3,
        n_passengers: int = 5,
        node_event_prob: float = 0.00,
        edge_event_prob: float = 0.00,
        max_steps: int = 1000,
        seed: int = None,
    ):
        self.grid_size = grid_size
        self.n_taxis = n_taxis
        self.n_passengers = n_passengers
        self.node_event_prob = node_event_prob
        self.edge_event_prob = edge_event_prob
        self.max_steps = max_steps

        self.rng = np.random.RandomState(seed)

        self.t = 0  # global time (for traffic light)
        self.step_count = 0

        # Taxi state
        self.taxi_pos = np.zeros((n_taxis, 2), dtype=int)
        self.taxi_has_passenger = np.zeros(n_taxis, dtype=bool)
        self.taxi_target_pid = np.full(n_taxis, -1, dtype=int)  # 正在服務的乘客 id，-1 表示沒有

        # Passengers
        self.passenger_pos = np.zeros((n_passengers, 2), dtype=int)
        self.passenger_dest = np.zeros((n_passengers, 2), dtype=int)
        self.passenger_onboard = np.zeros(n_passengers, dtype=bool)
        self.passenger_done = np.zeros(n_passengers, dtype=bool)

        # Traffic lights: 每個 node 都有一個周期（紅綠燈），但我們先簡化成：
        #  - 只要 (x + y) % 2 == 0 -> cycle 4 (綠 2 步, 紅 2 步)
        #  - 否則 cycle 6 (綠 3 步, 紅 3 步)
        self.light_cycle = np.zeros((grid_size, grid_size), dtype=int)
        for x in range(grid_size):
            for y in range(grid_size):
                if (x + y) % 2 == 0:
                    self.light_cycle[x, y] = 4
                else:
                    self.light_cycle[x, y] = 6

        # Node / edge events
        self.node_blocked = np.zeros((grid_size, grid_size), dtype=bool)
        # Edge: 垂直 / 水平，使用兩個 boolean array
        # vertical edges: from (x,y) to (x+1,y) for x in [0, grid_size-2]
        self.edge_blocked_v = np.zeros((grid_size - 1, grid_size), dtype=bool)
        # horizontal edges: from (x,y) to (x,y+1) for y in [0, grid_size-2]
        self.edge_blocked_h = np.zeros((grid_size, grid_size - 1), dtype=bool)

        # episode 統計
        self.episode_completed = 0
        self.episode_collisions = 0

        # init
        self.reset()

    # ------------------ 主要 API：reset, step, render ------------------

    def reset(self):
        self.t = 0
        self.step_count = 0

        self.episode_completed = 0
        self.episode_collisions = 0

        # init taxis
        self.taxi_pos = self._random_free_positions(self.n_taxis)

        # init passengers
        self.passenger_pos = self._random_free_positions(self.n_passengers)
        self.passenger_dest = self._random_free_positions(self.n_passengers)
        self.passenger_onboard[:] = False
        self.passenger_done[:] = False

        self.taxi_has_passenger[:] = False
        self.taxi_target_pid[:] = -1

        self.node_blocked[:, :] = False
        self.edge_blocked_v[:, :] = False
        self.edge_blocked_h[:, :] = False

        obs = self._get_all_observations()
        return obs

    def step(self, actions: List[int]):
        """
        actions: list of int, len = n_taxis
        回傳: obs, rewards, done, info
        """
        assert len(actions) == self.n_taxis

        self.step_count += 1
        self.t += 1

        rewards = np.zeros(self.n_taxis, dtype=float)

        prev_positions = self.taxi_pos.copy()

        # 先處理 move 類動作，避免同一 step pick/drop 先後順序太多細節
        proposed_positions = self.taxi_pos.copy()

        for i, a in enumerate(actions):
            if a in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
                new_pos, move_penalty = self._attempt_move(i, a)
                proposed_positions[i] = new_pos
                rewards[i] += move_penalty
            elif a == ACTION_STAY:
                rewards[i] += -0.01  # 少量 time cost，避免一直原地不動
            # pickup/dropoff 待會再處理

        # 檢查碰撞（node 上兩台車）
        collision_mask = self._check_collisions(proposed_positions)
        if np.any(collision_mask):
            # 撞車：現在只給罰分，不直接終止 episode
            for idx in np.where(collision_mask)[0]:
                rewards[idx] += -10.0
            self.episode_collisions += 1

        # 更新位置（即使有碰撞，也先往 new_pos 移動）
        self.taxi_pos = proposed_positions

        # ⭐ 距離型 reward shaping（收斂版，不要壓過 main reward）
        for i in range(self.n_taxis):
            prev_x, prev_y = prev_positions[i]
            new_x, new_y = self.taxi_pos[i]

            # 僅針對「目前的目標」（沒目標就不 shaping）
            tx, ty = self._get_taxi_target_pos(i)
            if tx is None:
                continue

            prev_dist = abs(prev_x - tx) + abs(prev_y - ty)
            new_dist = abs(new_x - tx) + abs(new_y - ty)

            # 靠近目標給一點小鼓勵，遠離給一點小懲罰
            if new_dist < prev_dist:
                rewards[i] += 0.02
            elif new_dist > prev_dist:
                rewards[i] -= 0.02

        # ⭐ pickup / dropoff（注意：在上面 for 迴圈外）
        drop_success = False

        for i, a in enumerate(actions):
            if a == ACTION_PICKUP:
                rewards[i] += self._attempt_pickup(i)

            elif a == ACTION_DROPOFF:
                r = self._attempt_dropoff(i)
                if r > 0:
                    drop_success = True
                rewards[i] += r

        # 如果有任何一台成功下客，大家一起加一點 team bonus
        if drop_success:
            rewards += 5.0
            self.episode_completed += 1

        # 每一步的時間成本（整體一點點 time pressure）
        rewards += -0.02

        # 更新事件（可以設計成每幾步更新一次，這裡先每一步都抽）
        self._sample_new_events()

        # 終了條件：所有乘客都完成 or step 達上限
        all_done = np.all(self.passenger_done)
        timeout = self.step_count >= self.max_steps
        done = bool(all_done or timeout)

        obs = self._get_all_observations()
        info = {
            "all_passengers_done": bool(all_done),
            "timeout": bool(timeout),
            "episode_completed": int(self.episode_completed),
            "episode_collisions": int(self.episode_collisions),
        }

        return obs, rewards, done, info

    # ------------------------------------------------
    # 內部實作細節
    # ------------------------------------------------
    def _random_free_positions(self, n: int) -> np.ndarray:
        """
        在 grid 上隨機選 n 個 node 當作初始位置。
        （這裡沒有特別避開 taxi / passenger 重疊，有需要可以加 constraint）
        """
        xs = self.rng.randint(0, self.grid_size, size=n)
        ys = self.rng.randint(0, self.grid_size, size=n)
        return np.stack([xs, ys], axis=1)

    def _get_all_observations(self) -> np.ndarray:
        """
        每台 taxi 的 observation：
        - 自己的位置 (x, y) / grid_size 正規化
        - 是否載客 flag
        - 目標乘客相對位置 / grid_size（若無目標則 0）
        - 附近紅綠燈狀態（簡化：當前 node 的紅綠燈）
        """
        obs_list = []
        for i in range(self.n_taxis):
            obs_list.append(self._get_observation_for_taxi(i))
        return np.stack(obs_list, axis=0)

    def _get_observation_for_taxi(self, taxi_id: int) -> np.ndarray:
        x, y = self.taxi_pos[taxi_id]
        has_p = float(self.taxi_has_passenger[taxi_id])

        # 目標點（若有載客 -> 目的地；若沒載客 -> 最近的乘客）
        tx, ty = self._get_taxi_target_pos(taxi_id)

        if tx is None:
            tx = x
            ty = y

        dx = (tx - x) / self.grid_size
        dy = (ty - y) / self.grid_size

        # 紅綠燈狀態（1=綠,0=紅）
        light_green = float(self._is_light_green(x, y))

        # 當前 grid 上乘客比例（尚未完成 & 未上車）
        remaining_passengers = np.sum(~self.passenger_done)
        passenger_fraction = remaining_passengers / max(1, self.n_passengers)

        obs = np.array(
            [
                x / self.grid_size,
                y / self.grid_size,
                has_p,
                dx,
                dy,
                light_green,
                passenger_fraction,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_taxi_target_pos(self, taxi_id: int) -> Tuple[int, int] | Tuple[None, None]:
        """
        若該 taxi 已經載客，則目標是該乘客的目的地。
        否則目標是「最近的尚未完成且未上車的乘客」。
        若場上沒有可服務的乘客則回傳 (None, None)。
        """
        if self.taxi_has_passenger[taxi_id]:
            pid = self.taxi_target_pid[taxi_id]
            if pid < 0:
                return None, None
            if self.passenger_done[pid]:
                return None, None
            dx, dy = self.passenger_dest[pid]
            return int(dx), int(dy)

        # 沒載客 -> 找最近的乘客
        x, y = self.taxi_pos[taxi_id]
        best_pid = -1
        best_dist = 1e9
        for pid in range(self.n_passengers):
            if self.passenger_done[pid] or self.passenger_onboard[pid]:
                continue
            px, py = self.passenger_pos[pid]
            dist = abs(px - x) + abs(py - y)
            if dist < best_dist:
                best_dist = dist
                best_pid = pid

        if best_pid < 0:
            return None, None
        px, py = self.passenger_pos[best_pid]
        return int(px), int(py)

    # ------------------ pickup / dropoff ------------------
    def _attempt_pickup(self, taxi_id: int) -> float:
        """
        試著在當前位置接客，成功 +10
        """
        if self.taxi_has_passenger[taxi_id]:
            # 已經載人不能再接
            return -1.0

        x, y = self.taxi_pos[taxi_id]
        reward = -0.5  # 無效操作的小罰分

        for pid in range(self.n_passengers):
            if self.passenger_done[pid] or self.passenger_onboard[pid]:
                continue
            px, py = self.passenger_pos[pid]
            if px == x and py == y:
                # 成功接客
                self.taxi_has_passenger[taxi_id] = True
                self.taxi_target_pid[taxi_id] = pid
                self.passenger_onboard[pid] = True
                reward = 30.0
                break

        return reward

    def _attempt_dropoff(self, taxi_id: int) -> float:
        """
        試著在當前位置下客，成功 +30
        """
        if not self.taxi_has_passenger[taxi_id]:
            return -1.0  # 沒載人卻 dropoff

        pid = self.taxi_target_pid[taxi_id]
        if pid < 0:
            return -1.0

        x, y = self.taxi_pos[taxi_id]
        dx, dy = self.passenger_dest[pid]

        if x == dx and y == dy:
            # 成功送達
            self.taxi_has_passenger[taxi_id] = False
            self.taxi_target_pid[taxi_id] = -1
            self.passenger_onboard[pid] = False
            self.passenger_done[pid] = True
            self.passenger_pos[pid] = np.array([-1, -1])
            return 200.0
        else:
            # 位置錯誤
            return -1.0

    # ------------------ move & collision ------------------
    def _attempt_move(self, taxi_id: int, action: int) -> Tuple[np.ndarray, float]:
        """
        移動 taxi，回傳 (new_pos, reward_penalty)
        """
        x, y = self.taxi_pos[taxi_id]
        new_x, new_y = x, y

        if action == ACTION_UP:
            new_x = max(0, x - 1)
        elif action == ACTION_DOWN:
            new_x = min(self.grid_size - 1, x + 1)
        elif action == ACTION_LEFT:
            new_y = max(0, y - 1)
        elif action == ACTION_RIGHT:
            new_y = min(self.grid_size - 1, y + 1)

        # 如果位置沒變（原本在邊界）就當 stay
        if new_x == x and new_y == y:
            return np.array([x, y]), -0.02  # 碰到邊界很小的罰分

        # 檢查 edge 是否被封
        if not self._is_edge_traversable(x, y, new_x, new_y):
            # edge 被封路，不能移動（但不至於超爆炸）
            return np.array([x, y]), -1.0

        # 檢查目標 node 是否被封
        if self.node_blocked[new_x, new_y]:
            return np.array([x, y]), -1.0

        # 檢查紅綠燈（簡化：若目標 node 是 red，move 仍會做，但給罰分；
        # 你也可以改成直接禁止進入）
        reward = 0.0
        if not self._is_light_green(new_x, new_y):
            reward -= 0.1  # 闖紅燈懲罰：存在但不會壓過主要任務

        return np.array([new_x, new_y]), reward

    def _is_edge_traversable(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        檢查從 (x1,y1) 到 (x2,y2) 的 edge 是否被封路
        只允許 Manhattan move (一格)
        """
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) + abs(dy) != 1:
            # 非合法移動（應該不會發生）
            return False

        # 水平 edge
        if dx == 1 and dy == 0:
            # vertical edge from (x1,y1) to (x1+1,y1)
            return not self.edge_blocked_v[x1, y1]
        if dx == -1 and dy == 0:
            return not self.edge_blocked_v[x2, y2]

        # 垂直 edge
        if dx == 0 and dy == 1:
            return not self.edge_blocked_h[x1, y1]
        if dx == 0 and dy == -1:
            return not self.edge_blocked_h[x2, y2]

        return True

    def _check_collisions(self, proposed_positions: np.ndarray) -> np.ndarray:
        """
        簡單檢查：若兩台 taxi 最後落在同一個 node，視為碰撞
        """
        collision = np.zeros(self.n_taxis, dtype=bool)
        for i in range(self.n_taxis):
            for j in range(i + 1, self.n_taxis):
                if (
                    proposed_positions[i, 0] == proposed_positions[j, 0]
                    and proposed_positions[i, 1] == proposed_positions[j, 1]
                ):
                    collision[i] = True
                    collision[j] = True
        return collision

    # ------------------ traffic lights & events ------------------
    def _is_light_green(self, x: int, y: int) -> bool:
        """
        簡單模型：
        每個 node 有一個 cycle = self.light_cycle[x,y]
        假設 cycle / 2 步是綠燈，cycle / 2 步是紅燈
        """
        cycle = self.light_cycle[x, y]
        if cycle <= 0:
            return True
        phase = self.t % cycle
        return phase < (cycle // 2)  # 前半段為綠燈

    def _sample_new_events(self):
        """
        每一步都有機率重新 random 一些 node / edge 被封路。
        目前設計很簡單：
          - 每個 node 以 node_event_prob 的機率變成 blocked
          - 每個 edge 以 edge_event_prob 的機率變成 blocked
        你可以改成：事件一旦發生會持續幾個 step 再消失（需要額外狀態）。
        """
        # reset
        self.node_blocked[:, :] = False
        self.edge_blocked_v[:, :] = False
        self.edge_blocked_h[:, :] = False

        # nodes
        mask_node = self.rng.rand(self.grid_size, self.grid_size) < self.node_event_prob
        self.node_blocked[mask_node] = True

        # vertical edges
        mask_ev = self.rng.rand(self.grid_size - 1, self.grid_size) < self.edge_event_prob
        self.edge_blocked_v[mask_ev] = True

        # horizontal edges
        mask_eh = self.rng.rand(self.grid_size, self.grid_size - 1) < self.edge_event_prob
        self.edge_blocked_h[mask_eh] = True

    # ------------------ render ------------------
    def render_ascii(self, show_passengers=True):
        """
        在 terminal 裡粗略看一下狀態：
        - '.' = 空
        - 'X' = node blocked
        - 'P' = passenger（未完成 & 未上車）
        - 'T','U','V' = 三台 taxi
        """
        g = np.full((self.grid_size, self.grid_size), '.', dtype='<U2')

        # 畫 blocked nodes
        bx, by = np.where(self.node_blocked)
        g[bx, by] = 'X'

        # 畫 passengers
        if show_passengers:
            for pid in range(self.n_passengers):
                if self.passenger_done[pid]:
                    continue
                if self.passenger_onboard[pid]:
                    continue
                px, py = self.passenger_pos[pid]
                if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                    g[px, py] = 'P'

        # 畫 taxis
        taxi_chars = ['T', 'U', 'V', 'W', 'Y', 'Z']
        for i in range(self.n_taxis):
            x, y = self.taxi_pos[i]
            ch = taxi_chars[i % len(taxi_chars)]
            g[x, y] = ch

        for x in range(self.grid_size):
            print(''.join(g[x, :]))

    def render_matplotlib(self):
        """
        用 matplotlib 大致畫出 grid、taxi、乘客、blocked nodes。
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.grid(True)

        # blocked nodes
        bx, by = np.where(self.node_blocked)
        ax.scatter(by, bx, c='red', marker='s', label='Blocked')

        # passengers
        for pid in range(self.n_passengers):
            if self.passenger_done[pid]:
                continue
            if self.passenger_onboard[pid]:
                continue
            px, py = self.passenger_pos[pid]
            ax.scatter(py, px, c='blue', marker='o')

        # taxis
        colors = ['green', 'orange', 'purple', 'black']
        for i in range(self.n_taxis):
            x, y = self.taxi_pos[i]
            ax.scatter(y, x, c=colors[i % len(colors)], marker='^')

        ax.invert_yaxis()
        plt.show()


# ------------------ MultiAgent Wrapper 給 PPO 用 ------------------

class MultiAgentWrapper:
    """
    把 RoboTaxiEnv 包成 multi-agent 形式，符合 train.py 裡的接口：

    - env.reset() 回傳 dict[agent_id] = obs (np.ndarray)
    - env.step(action_dict) 回傳 (obs_dict, reward_dict, done_dict, info)
    """

    def __init__(self, base_env: RoboTaxiEnv):
        self.base_env = base_env
        self.agents = [f"taxi_{i}" for i in range(base_env.n_taxis)]

    def reset(self) -> Dict[str, np.ndarray]:
        obs_arr = self.base_env.reset()  # shape: (n_taxis, obs_dim)
        obs_dict = {
            agent_id: obs_arr[i]
            for i, agent_id in enumerate(self.agents)
        }
        return obs_dict

    def step(
        self,
        action_dict: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict]:
        """
        action_dict: {agent_id: action}
        """
        actions = np.zeros(len(self.agents), dtype=int)
        for i, agent_id in enumerate(self.agents):
            actions[i] = action_dict.get(agent_id, ACTION_STAY)

        obs_arr, rewards_arr, done, info = self.base_env.step(actions)

        obs_dict = {
            agent_id: obs_arr[i]
            for i, agent_id in enumerate(self.agents)
        }
        reward_dict = {
            agent_id: float(rewards_arr[i])
            for i, agent_id in enumerate(self.agents)
        }
        done_dict = {
            agent_id: bool(done)
            for agent_id in self.agents
        }

        return obs_dict, reward_dict, done_dict, info


if __name__ == "__main__":
    # 簡單測一下 env 本身
    env = RoboTaxiEnv(grid_size=5, n_taxis=1, n_passengers=1, node_event_prob=0.0, edge_event_prob=0.0)
    obs = env.reset()
    print("Initial obs:", obs)

    for t in range(10):
        action = np.random.randint(0, 7, size=env.n_taxis)
        obs, rew, done, info = env.step(action)
        print(f"t={t}, action={action}, reward={rew}, done={done}")
        env.render_ascii()
        print()
        if done:
            break
