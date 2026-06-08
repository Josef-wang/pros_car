"""Auto Task Controller — 自動完成 Task 1/2/3 的狀態機。

每個 task 在背景執行緒跑一個狀態機，重用既有積木：
  - 車輪：car_controller.update_action(ACTION_KEY)
  - 視覺：data_processor.get_yolo_target_info() = [found, distance_m, delta_x_px]
  - 手臂：arm_controller.project_and_grab_from_depth() / release()
  - 回起點：car_controller.nav_processing 的 Nav2 p2p

由 mode_manager.AutoTaskMode 呼叫 run(task_name, key)：按 's' 啟動、'q' 中止。

重要前提：手臂沒有左右(yaw)自由度，橫向對齊一律由車輪完成 (FINE_ALIGN)。
"""

import math
import threading
import time


class AutoTaskController:
    def __init__(self, car_controller, arm_controller, data_processor, ros_communicator):
        self.car_controller = car_controller
        self.arm_controller = arm_controller
        self.data_processor = data_processor
        self.ros_communicator = ros_communicator
        self.nav_processing = car_controller.nav_processing

        # 執行緒管理
        self._thread = None
        self._stop_event = None
        self._running = False

        # ===== 可調參數 =====
        self.target_class = "bear"      # Task 1 目標 class
        self.align_coarse = 80.0        # APPROACH 對齊門檻 (px)：越小越「先轉正再前進」，
                                        # 把熊壓在畫面中央、避免接近時滑出視野 (原 200 太鬆)
        self.align_fine = 40.0          # FINE_ALIGN 細對齊門檻 (px)
        self.stop_distance = 0.55       # 觸發停車進入細對齊的距離 (m)
                                        # 要比「熊掉出畫面的距離」稍大，車才會在熊還看得到時
                                        # 就停下做 FINE_ALIGN 旋轉對正，而非衝太近盲抓而偏掉
        self.observe_seconds = 5.0      # OBSERVE 靜止觀察秒數 (計分需求)
        # 掉幀寬限：目標在畫面邊緣時 bbox 會閃爍 (found 時有時無)。
        # 短暫掉幀時不立刻退回 SEARCH，先朝「最後已知方向」續轉 lost_grace 秒咬住它。
        self.lost_grace = 0.8           # 秒
        # 近距離跟丟 → 直接盲抓：熊靠太近會掉出畫面下緣，YOLO 可能轉而回報遠處別隻熊
        # (距離暴增) 或 not-found。若原鎖定的熊跟丟前「夠近且夠正」，就停穩盲抓，不追遠熊。
        self.grasp_lost_distance = 0.55  # 跟丟前距離 ≤ 此值才視為「近到可直接抓」(m)
        self.grasp_lost_center_px = 80.0  # 跟丟前 |delta_x| ≤ 此值才夠正、可盲抓 (px)
                                          # 收緊：盲抓只在熊夠正時才觸發，否則回 SEARCH 重來，避免偏掉
        self.jump_distance_thresh = 0.6  # 距離較上次有效值暴增超過此量 → 判定 YOLO 跳到遠熊 (m)
        self.close_lost_confirm = 0.4    # 近距離跟丟後停穩確認秒數 (濾掉瞬間閃爍) 才盲抓
        # 抓完是否「回起點並放下」(RETURN→RELEASE)。需要 localization_unity / AMCL。
        # 只測抓取時保持 False：GRASP 後直接 DONE，夾著不放。
        self.enable_return = True
        # RETURN 用 Nav2，距目標 <0.5m 就判定到達 → 車停在離起點半公尺處(場地側)、面朝起點。
        # ⚠️ 不要用 overshoot 把『回起點的 nav goal』推過頭：起點在場地邊緣靠牆，往車反方向
        #    過頭會把 goal 推到牆外/障礙裡，Nav2 規劃不出路徑 → 貼牆繞怪、接不到原點。
        #    RETURN 改用『到點後盲推』把熊送進區(車面朝起點，前進=進區，方向安全)。
        #    overshoot helper(_navigate_to)留給 Task2/3 在開闊區精準靠泊用。
        self.return_overshoot = 0.0     # 公尺，RETURN 一律 0 (見上)；helper 仍支援
        # 放開前的收尾：Nav2 在離起點 0.5m 處就停、且朝向常平行邊界 → 盲推也進不了區。
        # 改用『閉環朝起點補完最後一段』(_creep_to_point)：用 AMCL 回授轉向起點再前進，
        # 直到離起點 ≤ release_creep_tol 才放熊。方向永遠朝起點(=區內)，不靠 Nav2 停下時朝向。
        self.release_creep_tol = 0.2    # 公尺：補到離起點這麼近才放 (太遠不進區→調小；衝過頭→調大)
        self.release_creep_timeout = 6.0  # 秒：閉環補位安全上限
        # 補位後額外的盲推前進秒數，預設 0(閉環已到位)；若還想把熊再往區內推一點再開
        self.release_creep_time = 0.0   # 秒
        # task 啟動時自動重發 /initialpose 把 AMCL 重定位於起點。
        # ⚠️ 前提：開 task 時車一定在 spawn 起點 (否則會把定位設錯)。
        # respawn 車後不必重開 localization。座標 = 起點 (0,0,0)。
        self.reanchor_on_start = True
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0

        # SEARCH 搜尋策略
        self.search_use_open_heuristic = True  # 用深度挑開闊側當起始旋轉方向
        self.search_base_sweep = 1.5    # 第一段掃描秒數
        self.search_sweep_increment = 1.0  # 每次反向後增加的掃描秒數 (擴張擺掃)
        self.search_unstick_backward = 0.3  # 反向時後退脫困秒數 (0 = 關閉)

        # 抓取參數
        self.grasp_mode = "fixed"       # "fixed" 或 "tf_depth"
        self.fixed_distance = 0.42      # fixed 模式的前方距離 (m)
        self.bear_height = 0.05         # 目標中心高度 (base_footprint, m)
        # 夾爪末端微調 (校正小誤差用，arm_ik_base 座標)
        self.grasp_reach_bias = 0.0     # +往前伸更多 / -往回收 (m)
        self.grasp_height_bias = 0.0    # +往上 / -往下 (m)
        # 抓取前的盲推：depth <40cm 會失效，車常停在手臂搆不到的距離。
        # 進 GRASP 前先盲推前進一小段把熊帶進可及範圍 (0 = 關閉)。
        self.final_creep_time = 0.6     # 秒 (依抓取結果調：壓過頭→減小、搆不到→加大)

    # ==========================================
    # 對外介面 (給 mode 呼叫)
    # ==========================================
    def run(self, task_name, key):
        """mode 的每次按鍵都會進來。按 's' 啟動、'q' 中止、'i' 重定位起點。"""
        if key == "q":
            self.stop()
            return True
        if key == "i":
            self.reanchor()
            return False
        if key == "s" and not self._running:
            self.start(task_name)
        return False

    def reanchor(self):
        """手動重發 /initialpose 把 AMCL 重定位於起點 (car respawn 後用，免重開 localization)。"""
        print(f"📍 重定位 AMCL 於起點 ({self.start_x}, {self.start_y}, yaw={self.start_yaw})")
        self.ros_communicator.publish_initial_pose(
            self.start_x, self.start_y, self.start_yaw
        )

    def start(self, task_name):
        if self._running:
            print("⚠️ 已有 task 在執行中。")
            return
        loop_fn = {
            "task1": self._task1_loop,
            # "task2": self._task2_loop,  # ⏳ 待實作
            # "task3": self._task3_loop,  # ⏳ 待實作
        }.get(task_name)
        if loop_fn is None:
            print(f"⚠️ {task_name} 尚未實作。")
            return
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=loop_fn, args=(self._stop_event,), daemon=True
        )
        self._thread.start()
        self._running = True
        print(f"▶️ {task_name} 啟動 (按 q 中止)。")

    def stop(self):
        if self._running and self._stop_event is not None:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        self._running = False
        self.car_controller.update_action("STOP")
        self.ros_communicator.publish_yolo_target_class("")  # 取消 class 過濾
        print("⏹️ Auto task 已中止。")

    # ==========================================
    # Task 1 狀態機
    # ==========================================
    def _task1_loop(self, stop_event):
        car = self.car_controller
        dp = self.data_processor
        arm = self.arm_controller

        # task 啟動時重發 initialpose 把 AMCL 重定位於起點 (car respawn 後免重開 localization)
        if self.reanchor_on_start:
            print(f"[Task1] 重發 initialpose 於起點 ({self.start_x}, {self.start_y})")
            self.ros_communicator.publish_initial_pose(
                self.start_x, self.start_y, self.start_yaw
            )
            time.sleep(0.5)  # 給 AMCL 一點時間吃掉新位姿再記起點

        # 只有開啟 RETURN 才需要起點 (避免 latched/殘留的 amcl_pose 誤觸 RELEASE)
        start_pose = self._get_start_pose() if self.enable_return else None
        self.ros_communicator.publish_yolo_target_class(self.target_class)

        state = "SEARCH"
        observe_start = None
        last_valid_depth = None
        # 目標追蹤記憶 (給掉幀寬限用)
        last_seen_delta_x = 0.0   # 目標最後出現時在左(-)還在右(+)
        lost_start = None         # 掉幀起始時刻；found 時清為 None
        close_lost_start = None   # 近距離跟丟確認計時起點
        # SEARCH 擺掃狀態 (search_dir=None 代表進入 SEARCH 時重新初始化)
        search_dir = None
        search_until = None
        search_duration = self.search_base_sweep
        print("[Task1] 狀態機啟動 → SEARCH")

        while not stop_event.is_set():
            now = time.time()
            info = dp.get_yolo_target_info()  # [found, distance, delta_x] or None
            found = info is not None and info[0] == 1.0
            distance = info[1] if info is not None else 0.0
            delta_x = info[2] if info is not None else 0.0
            # 距離較上次有效值暴增 → YOLO 把目標換成了遠處別隻熊，視同原目標跟丟
            jumped_far = (
                found
                and distance > 0.0
                and last_valid_depth is not None
                and distance > last_valid_depth + self.jump_distance_thresh
            )
            if found and not jumped_far:
                last_seen_delta_x = delta_x
                lost_start = None
                close_lost_start = None  # 重新看到目標 → 取消盲抓確認
                if distance > 0.0:
                    last_valid_depth = distance
            elif lost_start is None:
                lost_start = now
            # 原鎖定目標是否跟丟 (沒偵測到 或 被換成遠熊)
            near_lost = (not found) or jumped_far
            # 跟丟前是否「夠近 + 夠正」→ 可直接盲抓 (熊掉到畫面下緣，相機照不到)
            was_close = (
                last_valid_depth is not None
                and last_valid_depth <= self.grasp_lost_distance
            )
            was_centered = abs(last_seen_delta_x) <= self.grasp_lost_center_px
            # 掉幀仍在寬限內？(只在追蹤中的狀態使用)
            in_grace = lost_start is not None and (now - lost_start) < self.lost_grace
            # 跟丟時朝最後已知方向旋轉回找：在左→左轉、在右→右轉
            reacquire_rot = (
                "COUNTERCLOCKWISE_ROTATION_SLOW"
                if last_seen_delta_x < 0
                else "CLOCKWISE_ROTATION_SLOW"
            )

            prev_state = state

            if state == "SEARCH":
                if found:
                    car.update_action("STOP")
                    search_dir = None
                    state = "APPROACH"
                else:
                    if search_dir is None:
                        # 進入 SEARCH：挑開闊側起步、重設擺掃
                        search_dir = self._pick_open_direction()
                        search_duration = self.search_base_sweep
                        search_until = now + search_duration
                        print(f"[Task1] SEARCH 起始方向: {search_dir}")
                    elif now >= search_until:
                        # 反向 + 擴張掃描範圍 + 後退脫困
                        search_dir = "CW" if search_dir == "CCW" else "CCW"
                        search_duration += self.search_sweep_increment
                        search_until = now + search_duration
                        if self.search_unstick_backward > 0:
                            self._timed_action(
                                "BACKWARD_SLOW",
                                self.search_unstick_backward,
                                stop_event,
                            )
                        print(
                            f"[Task1] SEARCH 反向 → {search_dir}, 掃 {search_duration:.1f}s"
                        )
                    rot = (
                        "COUNTERCLOCKWISE_ROTATION_SLOW"
                        if search_dir == "CCW"
                        else "CLOCKWISE_ROTATION_SLOW"
                    )
                    car.update_action(rot)

            elif state == "APPROACH":
                if near_lost:
                    if was_close and was_centered:
                        # 近距離跟丟 + 夠正 → 熊在正前下方相機照不到，停穩確認後盲抓
                        car.update_action("STOP")
                        if close_lost_start is None:
                            close_lost_start = now
                        if now - close_lost_start >= self.close_lost_confirm:
                            print(
                                f"[Task1] 近距離跟丟(d≈{last_valid_depth:.2f}m) → 直接 GRASP"
                            )
                            close_lost_start = None
                            state = "GRASP"
                    elif in_grace:
                        # 短暫掉幀(還沒近/沒正)：朝最後已知方向續轉，把熊帶回中央
                        car.update_action(reacquire_rot)
                    else:
                        car.update_action("STOP")
                        search_dir = None
                        state = "SEARCH"
                elif delta_x > self.align_coarse:
                    car.update_action("CLOCKWISE_ROTATION_SLOW")        # 熊偏右 → 右轉對正
                elif delta_x < -self.align_coarse:
                    car.update_action("COUNTERCLOCKWISE_ROTATION_SLOW") # 熊偏左 → 左轉對正
                elif distance < 0.0 or (0.0 < distance <= self.stop_distance):
                    # 到達停車距離，或太近 depth=-1 視同到位
                    car.update_action("STOP")
                    state = "FINE_ALIGN"
                else:
                    car.update_action("FORWARD_SLOW")

            elif state == "FINE_ALIGN":
                if near_lost:
                    if was_close and was_centered:
                        car.update_action("STOP")
                        if close_lost_start is None:
                            close_lost_start = now
                        if now - close_lost_start >= self.close_lost_confirm:
                            print(
                                f"[Task1] 近距離跟丟(d≈{last_valid_depth:.2f}m) → 直接 GRASP"
                            )
                            close_lost_start = None
                            state = "GRASP"
                    elif in_grace:
                        car.update_action(reacquire_rot)
                    else:
                        car.update_action("STOP")
                        search_dir = None
                        state = "SEARCH"
                elif delta_x > self.align_fine:
                    car.update_action("CLOCKWISE_ROTATION_SLOW")        # 精對齊：右轉
                elif delta_x < -self.align_fine:
                    car.update_action("COUNTERCLOCKWISE_ROTATION_SLOW") # 精對齊：左轉
                else:
                    car.update_action("STOP")
                    observe_start = time.time()
                    state = "OBSERVE"

            elif state == "OBSERVE":
                car.update_action("STOP")
                if near_lost:
                    if in_grace:
                        pass  # 短暫掉幀：原地等，不重置 5 秒觀察計時
                    else:
                        observe_start = None
                        state = "APPROACH"  # 真的跟丟/跳遠，回 APPROACH(會判斷近距離盲抓)
                elif now - observe_start >= self.observe_seconds:
                    print("[Task1] 觀察滿 5 秒 ✅ → GRASP")
                    state = "GRASP"

            elif state == "GRASP":
                # 抓取前盲推：把熊帶進手臂可及範圍 (depth 此時多半已失效)
                if self.final_creep_time > 0:
                    print(f"[Task1] GRASP 前盲推前進 {self.final_creep_time:.1f}s")
                    self._timed_action("FORWARD_SLOW", self.final_creep_time, stop_event)
                car.update_action("STOP")
                ok = arm.project_and_grab_from_depth(
                    depth=last_valid_depth,
                    mode=self.grasp_mode,
                    fixed_distance=self.fixed_distance,
                    bear_height=self.bear_height,
                    reach_bias=self.grasp_reach_bias,
                    height_bias=self.grasp_height_bias,
                )
                if not ok:
                    print(
                        "[Task1] ⚠️ 抓取失敗 (TF/投影問題)。"
                        "請確認 robot_state_publisher 有在跑 "
                        "(slam_unity.sh 或 docker-compose_robot_unity.yml)。中止。"
                    )
                    break
                state = "RETURN" if start_pose is not None else "DONE"

            elif state == "RETURN":
                self._return_to_start(start_pose, stop_event)
                state = "RELEASE"

            elif state == "RELEASE":
                # Nav2 停在離起點 0.5m 處(常停在計分區邊界外、朝向平行邊界) →
                # 閉環朝起點補完最後一段，把車(連同前方的熊)帶進區內再放。
                self._creep_to_point(
                    [start_pose[0], start_pose[1]], stop_event,
                    tol=self.release_creep_tol, timeout=self.release_creep_timeout,
                )
                if self.release_creep_time > 0:  # 可選：補位後再往區內推一點
                    print(f"[Task1] 放開前再盲推 {self.release_creep_time:.1f}s")
                    self._timed_action("FORWARD_SLOW", self.release_creep_time, stop_event)
                car.update_action("STOP")
                arm.release()
                state = "DONE"

            elif state == "DONE":
                car.update_action("STOP")
                print("[Task1] ✅ 完成。")
                break

            if state != prev_state:
                print(f"[Task1] {prev_state} → {state}")
                # 進入 SEARCH = 放棄當前目標 → 清掉目標記憶，避免舊 last_valid_depth
                # 讓之後找到的(較遠)熊一直被誤判 jumped_far，造成 SEARCH↔APPROACH 死迴圈
                if state == "SEARCH":
                    last_valid_depth = None
                    last_seen_delta_x = 0.0
                    lost_start = None
                    close_lost_start = None

            time.sleep(0.1)

        car.update_action("STOP")
        print("[Task1] 狀態機結束。")

    # ==========================================
    # 共用子程序
    # ==========================================
    def _pick_open_direction(self):
        """用 x_multi_depth 比較左右半邊開闊度，挑起始旋轉方向。
        左邊較開闊 → 'CCW'(左轉)；右邊較開闊 → 'CW'(右轉)；資料不足預設 'CCW'。
        牆那側深度小，因此會往比較空的一側轉，避免一開始就鑽牆。
        """
        if not self.search_use_open_heuristic:
            return "CCW"
        depths = self.data_processor.get_camera_x_multi_depth()
        if not depths:
            return "CCW"
        n = len(depths)
        left = [d for d in depths[: n // 2] if d > 0.0]
        right = [d for d in depths[n // 2 :] if d > 0.0]
        left_avg = sum(left) / len(left) if left else 0.0
        right_avg = sum(right) / len(right) if right else 0.0
        return "CCW" if left_avg >= right_avg else "CW"

    def _timed_action(self, action, duration, stop_event):
        """持續發某個動作指令 duration 秒 (可被 stop_event 中斷)。"""
        end = time.time() + duration
        while time.time() < end and not stop_event.is_set():
            self.car_controller.update_action(action)
            time.sleep(0.05)

    def _get_start_pose(self, timeout=5.0):
        """輪詢等待 /amcl_pose（AMCL 要先收到 initial pose + 一次更新才會發）。
        最多等 timeout 秒，仍拿不到才放棄 RETURN。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                pose, _ = self.data_processor.get_processed_amcl_pose()
                if pose is not None:
                    print(f"[Task1] 記錄起點: ({pose[0]:.2f}, {pose[1]:.2f})")
                    return pose
            except Exception:
                pass
            time.sleep(0.2)
        print(
            "⚠️ 等不到 /amcl_pose（已等 {:.0f}s），RETURN 將略過。".format(timeout)
            + "請確認已在 Foxglove 設過 initial pose，且 `ros2 topic hz /amcl_pose` 有在跳。"
        )
        return None

    def _return_to_start(self, start_pose, stop_event):
        print(f"[Task1] RETURN → 回起點 ({start_pose[0]:.2f}, {start_pose[1]:.2f})")
        self._navigate_to(
            [start_pose[0], start_pose[1]], stop_event, overshoot=self.return_overshoot
        )

    # ==========================================
    # 共用導航 helper (Task 1/2/3 通用)
    # ==========================================
    def _current_xy(self):
        """目前車輛 (x, y)（map frame）；拿不到回 None。"""
        try:
            pose, _ = self.data_processor.get_processed_amcl_pose()
            return [pose[0], pose[1]]
        except Exception:
            return None

    def _overshoot_goal(self, target, approach_from, overshoot):
        """把 target 沿『approach_from → target』方向再往前延伸 overshoot 公尺後回傳。

        用途：Nav2 p2p「離終點 <0.5m 就判定到達」會讓車停在目標前半公尺；
        把目標過頭 overshoot，停下來剛好落在真正的 target 上。
        Task2/3 精準靠泊 (橋頭中線、門前定位) 可重用同一個抵銷邏輯。
        approach_from 為 None、overshoot<=0、或起點與 target 重合時，原樣回傳 target。"""
        if overshoot <= 0.0 or approach_from is None:
            return [target[0], target[1]]
        dx = target[0] - approach_from[0]
        dy = target[1] - approach_from[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return [target[0], target[1]]
        ux, uy = dx / dist, dy / dist
        return [target[0] + ux * overshoot, target[1] + uy * overshoot]

    def _creep_to_point(self, target, stop_event, tol=0.2, timeout=6.0):
        """用 AMCL 回授閉環，把車開到 target [x,y] 的 tol 公尺內。

        補 Nav2 p2p 最後一段：p2p 在離目標 0.5m 就判定到達，車常停在計分區邊界外、
        且朝向平行邊界，盲推進不了區。這裡每幀讀 AMCL → 算到 target 的朝向誤差，
        先轉正再前進，方向永遠朝 target(=區內)。Task2/3 精準到點也可重用。
        timeout 為安全上限，避免 AMCL 抖動或卡住時無限轉。"""
        nav = self.nav_processing
        car = self.car_controller
        print(f"[Nav] 閉環補位 → ({target[0]:.2f}, {target[1]:.2f}) tol={tol:.2f}")
        t0 = time.time()
        while not stop_event.is_set() and (time.time() - t0) < timeout:
            try:
                pose, quat = self.data_processor.get_processed_amcl_pose()
            except Exception:
                break
            if pose is None or quat is None:
                break
            dist = math.hypot(target[0] - pose[0], target[1] - pose[1])
            if dist <= tol:
                break
            diff = nav.calculate_diff_angle(pose, quat, target[0], target[1])
            if -20.0 <= diff <= 20.0:
                action = "FORWARD_SLOW"
            elif diff < -20.0:
                action = "CLOCKWISE_ROTATION_SLOW"
            else:
                action = "COUNTERCLOCKWISE_ROTATION_SLOW"
            car.update_action(action)
            time.sleep(0.05)
        car.update_action("STOP")

    def _navigate_to(self, target, stop_event, overshoot=0.0):
        """用 Nav2 p2p 導航到 target [x, y]（map frame）。

        overshoot>0 時把目標沿『目前位置→target』方向過頭，抵銷到達容差，停得更準
        (見 _overshoot_goal)。逼近方向取呼叫當下的車輛位置，故會自動依回來角度修正。
        Task2/3 可直接重用此 helper 做點對點導航 / 精準靠泊。"""
        nav = self.nav_processing
        nav.reset_nav_process()
        goal = self._overshoot_goal(target, self._current_xy(), overshoot)
        print(f"[Nav] → ({goal[0]:.2f}, {goal[1]:.2f}) overshoot={overshoot:.2f}")
        while not stop_event.is_set():
            action = nav.get_action_from_nav2_plan_no_dynamic_p_2_p(goal_coordinates=goal)
            self.car_controller.update_action(action)
            if nav.get_finish_flag():
                nav.reset_nav_process()
                break
            time.sleep(0.05)
        self.car_controller.update_action("STOP")
