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

from .nav2_utils import get_yaw_from_quaternion


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
        # 抓取站距 depth 閉環：先前進到 depth ≤ grasp_standoff(或 depth 進盲區失效)才停，
        # 把 FINE_ALIGN 那個受雜訊/網路影響而會飄的停車點，收斂到一致的近站距，
        # 再交給下面的 final_creep_time 盲推最後一段(盲區 depth 照不到，只能盲推)。
        self.grasp_standoff = 0.35      # 公尺，閉環目標站距(設在盲區邊緣)。0=關閉，退回純盲推
        self.grasp_creep_timeout = 4.0  # 秒，depth 閉環安全上限
        self.final_creep_time = 0.8     # 秒，盲區內最後一段盲推 (壓過頭→減小、搆不到→加大)

        # ---- Task 3 (開門) ----
        # 起步路標：朝門邊兩熊前進把門帶進視野。門邊熊在遠處(>YOLO max_target_distance)，
        # 必須用 "bear:far" 後綴讓 YOLO 反過來挑「最遠」熊、並關掉距離閘門，
        # 否則近模式會把遠熊濾光 → 車找不到目標、不會往門開過去。
        self.landmark_class = "bear:far"
        self.task3_target_class = "knob"    # 門把 class (detection.pt 內建)
        # 實測：knob 要靠到「離遠方兩熊約 1m」才穩定偵測得到。故切換點設 1.0m：
        # 熊靠到 1m 就切 knob，此時 knob 已可見、且熊還穩穩在畫面裡(未到跟丟距離)，
        # SEARCH 一轉就咬到 knob。切太近(舊 0.6)熊會先掉出畫面、knob 也早就該切了。
        self.landmark_stop_distance = 1.0   # 公尺：路標熊靠到這麼近就切 knob (=knob 可偵測距離)
        # 場上有別的近熊(Task1 熊等)當雜訊。門邊兩熊在場地另一頭、起點時很遠。
        # 未鎖定前，距離 < 此門檻的熊一律當雜訊(旋轉略過)，只 commit「夠遠」的門邊熊。
        # 實測雜熊出現在 ~1m；門邊熊起點時更遠 → 設 1.5m 區隔。太小會誤鎖近熊；
        # 太大若門邊熊起點沒那麼遠會永遠 commit 不到(只能靠 timeout fallback)。
        self.landmark_min_distance = 1.5    # 公尺：低於此距離的熊視為近雜熊、不當門路標
        # commit 後排除近熊雜訊：近熊出現在畫面『極左/右邊緣』(實測 dx≈-290)，而門邊熊
        # 會被我們轉到中央。故 |dx| 超過此門檻的偵測視為邊緣雜訊/即將出框 → 不追、不據以
        # 切換，只維持直行。比『距離連續性』穩(不會被閃爍離群值錨死)。
        self.landmark_edge_px = 230         # px：|dx| 超過此值的偵測當邊緣雜訊忽略
        # 置中後鎖航向沿正前方直行的設定(不依賴連續追熊，靠 AMCL 閉環)。
        self.landmark_drive_timeout = 15.0  # 秒：鎖航向直行的安全上限
        self.landmark_blind_time = 4.0      # 秒：拿不到 AMCL 時改盲推前進的秒數
        # 到門前找 knob：knob 在兩熊中間，到位時常偏一邊或還差一點距離。
        # SEARCH 改左右擴張擺掃，且每次反向往前挪一點(近一點更好認)，累計設上限免撞門。
        self.knob_search_creep = 0.3        # 秒：SEARCH 每次反向往前挪的盲推秒數
        self.knob_search_max_creep = 2.0    # 秒：SEARCH 累計前挪上限
        # timeout 純安全網：連續慢速接近(從 ~4m 邊轉邊前進)要夠長，別在還沒到門前就誤切。
        self.landmark_timeout = 35.0        # 秒：朝路標前進的安全上限，逾時才 fallback 切 knob
        self.knob_standoff = 0.35           # 公尺：壓門前 depth 閉環收斂站距
        # (B) depth 停點收穩：要求連續 N 幀 ≤target 才算到位、連續 M 幀無 depth 才算進盲區停，
        #     避免單幀雜訊提早停 → 停點 run-to-run 一致 (Task1 維持預設 1/3 不受影響)。
        self.knob_depth_confirm = 2         # 幀：連續幾幀 depth≤target 才停
        self.knob_depth_blind = 5           # 幀：連續幾幀無有效 depth 才判定進盲區停
        # 開門動作 = 舉高 → 靠到最近 → 壓下 → 前推。手臂用 arm_ik_base 直接定姿(不投影)：
        #   x=正前方伸出量、z=高度(上正下負)，同 x 不同 z 即「由上往下壓」。reach=0.191m。
        self.knob_arm_forward = 0.13        # 公尺：夾爪前伸量(arm_ik_base x)。搆不到→加大、頂到門→減小
        self.knob_raise_z = 0.10            # 公尺：舉高姿態 z (把手上方，貼近時不撞把手)
        self.knob_press_z = -0.02           # 公尺：壓下姿態 z (把手高度，由上往下壓)
        # (A) depth <0.4m 失效有盲區 → 舉高後「靠到最近」改用 AMCL 位移推固定距離(不靠計時)，
        #     消掉馬達 ramp/摩擦/延遲造成的落點變異。看 [creep] live 位移調。
        self.knob_final_creep_dist = 0.10   # 公尺：舉高後靠到最近的位移量 (搆不到→加大、撞門→減小)
        # fallback：拿不到 AMCL 時才退回計時盲推 (設 0 則不盲推)。
        self.knob_final_creep_time = 1.0    # 秒：無 AMCL 時靠到最近的盲推秒數
        self.door_close_gripper = True      # True=閉合夾爪當壓桿；False=張開 (待實測)
        # 壓下不收手，直接全速 FORWARD(=手動 w)一路推開。全速約半速兩倍距離 → 時間要短。
        self.door_push_time = 6.0           # 秒：壓住全速前推開門的時間 (沒全開→加長、撞過頭→縮短)
        # 前推時手臂在 knob 高度上下來回「掃」，補 FINE_ALIGN 左右微誤差 → 提高壓到把手機率。
        self.door_swing = True              # True=前推時手臂上下擺動；False=固定壓住
        self.door_swing_amp = 0.06          # 公尺：從壓下 z 往上掃的幅度 (掃不到→加大、頂到門→減小)
        # 門推開後先全速後退脫離門口，再接導航回原點 (免卡在門上/與門框糾纏)。
        self.door_back_time = 4.0           # 秒：開門後全速後退的時間
        # 開門完成後回起點 (需 localization_unity/AMCL + Nav2)。重用 Task1 回程積木，終點=起點(0,0)。
        self.task3_return_home = True

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
            "task3": self._task3_loop,
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
                # 1) depth 閉環收斂站距：消除 FINE_ALIGN 受雜訊/網路影響而飄的停車點，
                #    不管剛剛停在 0.5 還 0.65，都先開到一致的近站距(或 depth 進盲區)。
                if self.grasp_standoff > 0:
                    d = self._creep_to_depth(
                        self.grasp_standoff, stop_event, timeout=self.grasp_creep_timeout
                    )
                    if d is not None:
                        last_valid_depth = d  # 給投影用最新有效 depth
                # 2) 盲區內最後一段：depth 照不到，只能固定盲推把熊帶進手臂可及範圍
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
                # 回正車頭：_creep_to_point 只保證位置，朝向會指著起點(貼牆)方向。
                # 轉回 start_yaw(=spawn 朝向，面向場內)，讓車停回 home 姿態，
                # 後續 Task2/3「從起點直行」才走得出去。
                self._orient_to_yaw(self.start_yaw, stop_event)
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
    # Task 3 狀態機 (開門)
    # ==========================================
    def _task3_loop(self, stop_event):
        """門：朝兩熊路標前進 → 找門把(knob) → 對齊 → 觀察 5s →
        夾爪下壓門把 → 車前推開門。重用 Task1 的對齊/觀察/depth 閉環積木。
        計分：OBSERVE=Locate&Observe、PRESS+PUSH=Unlock+Clear。不需回起點。"""
        car = self.car_controller
        dp = self.data_processor
        arm = self.arm_controller

        if self.reanchor_on_start:
            print(f"[Task3] 重發 initialpose 於起點 ({self.start_x}, {self.start_y})")
            self.ros_communicator.publish_initial_pose(
                self.start_x, self.start_y, self.start_yaw
            )
            time.sleep(0.5)

        state = "APPROACH_LANDMARK"
        self.ros_communicator.publish_yolo_target_class(self.landmark_class)
        print("[Task3] 狀態機啟動 → APPROACH_LANDMARK (朝兩熊前進)")
        landmark_t0 = time.time()
        observe_start = None
        dbg_last = 0.0  # 診斷節流
        landmark_committed = False  # 是否已鎖定門邊遠熊(排除近雜熊後才 True)
        landmark_dist = None        # 信任的門邊熊距離(持續更新，平滑跟到 ~1m)
        last_seen_dx = 0.0          # 門邊熊最後已知橫向位置(跟丟時朝此方向續走)
        # SEARCH 擺掃狀態(找 knob 用)
        search_dir = None
        search_until = None
        search_dur = self.search_base_sweep
        knob_creep_used = 0.0       # SEARCH 階段累計前挪秒數

        while not stop_event.is_set():
            now = time.time()
            info = dp.get_yolo_target_info()
            found = info is not None and info[0] == 1.0
            distance = info[1] if info is not None else 0.0
            delta_x = info[2] if info is not None else 0.0
            prev_state = state

            # 診斷：每 ~1s 印一次目前 state + YOLO 回報，方便看「卡在哪、看到什麼」
            if now - dbg_last >= 1.0:
                dbg_last = now
                ld = f"{landmark_dist:.2f}" if landmark_dist is not None else "-"
                print(
                    f"[Task3][dbg] state={state} found={int(found)} "
                    f"dist={distance:.2f} dx={delta_x:.0f} "
                    f"committed={int(landmark_committed)} ldist={ld}"
                )

            if state == "APPROACH_LANDMARK":
                # 朝門邊『遠處兩熊』當路標把門帶進視野，全程連續追蹤+持續置中(每幀修正航向，
                # 不再一次置中後盲開→不會飄到熊側邊)。容忍 bbox 閃爍：
                #   - 近雜熊靠 YOLO node 的 far-lock + 邊緣濾除(|dx|>edge)雙重擋掉，不誤判成目標。
                #   - 短暫跟丟(found=0)不退狀態，朝『最後已知方向』續走咬回來(類 Task1 lost_grace)。
                # 持續更新 landmark_dist(信任值平滑跟到 ~1m)；到 ~1m 且置中 → 切 knob 交棒。
                far_enough = found and distance >= self.landmark_min_distance
                on_edge = found and abs(delta_x) > self.landmark_edge_px
                good = found and distance > 0.0 and not on_edge  # 信任的門邊熊幀
                timed_out = (now - landmark_t0) >= self.landmark_timeout

                if not landmark_committed:
                    if far_enough:
                        landmark_committed = True
                        landmark_dist = distance
                        last_seen_dx = delta_x
                        print(f"[Task3] 鎖定門邊遠熊 (dist={distance:.2f}m) → 連續追蹤接近")
                    elif not timed_out:
                        car.update_action("COUNTERCLOCKWISE_ROTATION_SLOW")  # 旋轉掃描找門邊熊
                else:
                    if good:
                        landmark_dist = distance     # 持續更新信任值(平滑跟到 ~1m，不凍結)
                        last_seen_dx = delta_x
                    reached = (
                        good and 0.0 < distance <= self.landmark_stop_distance
                        and abs(delta_x) <= self.align_coarse
                    )
                    if reached or timed_out:
                        if timed_out:
                            print("[Task3] APPROACH_LANDMARK 逾時 → 切 knob")
                        else:
                            print(f"[Task3] 到門前 (dist={distance:.2f}m,置中) → 切 knob")
                        car.update_action("STOP")
                        self.ros_communicator.publish_yolo_target_class(self.task3_target_class)
                        time.sleep(0.5)  # 等 YOLO 刷新成 knob，避免讀到殘留熊讀數
                        state = "SEARCH"
                    elif good and delta_x > self.align_coarse:
                        car.update_action("CLOCKWISE_ROTATION_SLOW")        # 偏右 → 右轉置中
                    elif good and delta_x < -self.align_coarse:
                        car.update_action("COUNTERCLOCKWISE_ROTATION_SLOW") # 偏左 → 左轉置中
                    elif good:
                        car.update_action("FORWARD_SLOW")                   # 已置中 → 前進接近
                    else:
                        # 跟丟/邊緣雜訊幀：不退狀態，朝最後已知方向續走咬回來
                        if abs(last_seen_dx) > self.align_coarse:
                            car.update_action(
                                "CLOCKWISE_ROTATION_SLOW" if last_seen_dx > 0
                                else "COUNTERCLOCKWISE_ROTATION_SLOW"
                            )
                        else:
                            car.update_action("FORWARD_SLOW")               # 上次已對正 → 續直行

            elif state == "SEARCH":
                if found:
                    car.update_action("STOP")
                    search_dir = None  # 重置擺掃狀態
                    state = "APPROACH"
                else:
                    # 左右擴張擺掃找 knob；每次反向往前挪一點(累計上限內)讓 knob 更近更好認
                    if search_dir is None:
                        search_dir = "CCW"
                        search_dur = self.search_base_sweep
                        search_until = now + search_dur
                    elif now >= search_until:
                        search_dir = "CW" if search_dir == "CCW" else "CCW"
                        search_dur += self.search_sweep_increment
                        search_until = now + search_dur
                        if knob_creep_used < self.knob_search_max_creep:
                            self._timed_action("FORWARD_SLOW", self.knob_search_creep, stop_event)
                            knob_creep_used += self.knob_search_creep
                            print(f"[Task3] SEARCH 找不到 knob，前挪 {self.knob_search_creep:.1f}s "
                                  f"(累計 {knob_creep_used:.1f}s)")
                    rot = (
                        "COUNTERCLOCKWISE_ROTATION_SLOW" if search_dir == "CCW"
                        else "CLOCKWISE_ROTATION_SLOW"
                    )
                    car.update_action(rot)

            elif state == "APPROACH":
                if not found:
                    state = "SEARCH"
                elif delta_x > self.align_coarse:
                    car.update_action("CLOCKWISE_ROTATION_SLOW")
                elif delta_x < -self.align_coarse:
                    car.update_action("COUNTERCLOCKWISE_ROTATION_SLOW")
                elif distance < 0.0 or (0.0 < distance <= self.stop_distance):
                    car.update_action("STOP")
                    state = "FINE_ALIGN"
                else:
                    car.update_action("FORWARD_SLOW")

            elif state == "FINE_ALIGN":
                if not found:
                    state = "APPROACH"
                elif delta_x > self.align_fine:
                    car.update_action("CLOCKWISE_ROTATION_SLOW")
                elif delta_x < -self.align_fine:
                    car.update_action("COUNTERCLOCKWISE_ROTATION_SLOW")
                else:
                    car.update_action("STOP")
                    observe_start = now
                    state = "OBSERVE"

            elif state == "OBSERVE":
                car.update_action("STOP")
                if observe_start is None:
                    observe_start = now
                elif now - observe_start >= self.observe_seconds:
                    print("[Task3] 觀察滿 5 秒 ✅ → PRESS")
                    state = "PRESS"

            elif state == "PRESS":
                # 開門序列：舉高 → 靠到最近 → 壓下 (前推在 PUSH)。
                # 1) (B) depth 閉環收斂一致站距：連續確認停點，消停車變異
                if self.knob_standoff > 0:
                    self._creep_to_depth(
                        self.knob_standoff, stop_event,
                        timeout=self.grasp_creep_timeout,
                        confirm_frames=self.knob_depth_confirm,
                        blind_frames=self.knob_depth_blind,
                    )
                car.update_action("STOP")
                # 2) 手伸直舉高 (夾爪閉合當壓桿，舉到把手上方)
                arm.arm_to_xz(
                    self.knob_arm_forward, self.knob_raise_z,
                    close_gripper=self.door_close_gripper, label="舉高",
                )
                # 3) (A) 靠到最近：用 AMCL 位移盲推固定距離 (取代計時、消推進變異)
                if self.knob_final_creep_dist > 0:
                    self._creep_forward_dist(
                        self.knob_final_creep_dist, stop_event,
                        timeout=self.grasp_creep_timeout,
                        fallback_time=self.knob_final_creep_time,
                    )
                car.update_action("STOP")
                # 4) 手臂壓下 (同 x、z 降到把手高度，由上往下壓並保持)
                arm.arm_to_xz(self.knob_arm_forward, self.knob_press_z, label="壓下")
                state = "PUSH"

            elif state == "PUSH":
                # 手臂維持下壓(不收回)，直接全速 FORWARD(=手動 w 的力道)一路前推開門。
                # door_swing：前推同時讓手臂在 knob 高度上下來回掃，補 FINE_ALIGN 左右微誤差。
                print(f"[Task3] 壓住全速前推開門 {self.door_push_time:.1f}s"
                      + ("，手臂上下擺動" if self.door_swing else ""))
                swing_stop = None
                swing_thr = None
                if self.door_swing and self.door_swing_amp > 0:
                    swing_stop = threading.Event()
                    swing_thr = threading.Thread(
                        target=self._arm_swing_loop,
                        args=(self.knob_arm_forward, self.knob_press_z,
                              self.knob_press_z + self.door_swing_amp, swing_stop),
                        daemon=True,
                    )
                    swing_thr.start()
                self._timed_action("FORWARD", self.door_push_time, stop_event)
                car.update_action("STOP")
                if swing_stop is not None:
                    swing_stop.set()
                    swing_thr.join(timeout=2.0)
                arm.reset_arm()                 # 開完才收手，免拖門/擋回程
                # 全速後退脫離門口，再接導航回原點
                if self.door_back_time > 0:
                    print(f"[Task3] 全速後退脫離門口 {self.door_back_time:.1f}s")
                    self._timed_action("BACKWARD", self.door_back_time, stop_event)
                    car.update_action("STOP")
                state = "RETURN" if self.task3_return_home else "DONE"

            elif state == "RETURN":
                # 回起點 (門已開，Nav2 可規劃穿門回原點)。需 AMCL，拿不到就略過。
                if self._current_xy() is None:
                    print("[Task3] ⚠️ 拿不到 AMCL，略過回起點 (需 localization_unity)。")
                else:
                    print(f"[Task3] 回起點 ({self.start_x:.2f}, {self.start_y:.2f})")
                    self._navigate_to(
                        [self.start_x, self.start_y], stop_event, overshoot=0.0
                    )
                    # Nav2 0.5m 容差 → 閉環補完最後一段回到原點
                    self._creep_to_point(
                        [self.start_x, self.start_y], stop_event,
                        tol=self.release_creep_tol, timeout=self.release_creep_timeout,
                    )
                    car.update_action("STOP")
                    # 同 Task1：轉回 start_yaw(spawn 朝向)，讓車頭回到起始姿態
                    self._orient_to_yaw(self.start_yaw, stop_event)
                    car.update_action("STOP")
                state = "DONE"

            elif state == "DONE":
                car.update_action("STOP")
                print("[Task3] ✅ 完成。")
                break

            if state != prev_state:
                print(f"[Task3] {prev_state} → {state}")

            time.sleep(0.1)

        car.update_action("STOP")
        print("[Task3] 狀態機結束。")

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

    def _drive_straight_ahead(self, bear_dist, stop_event):
        """置中門邊熊後，鎖航向沿『目前車頭正前方』直行到離熊約 landmark_stop_distance 處。

        Task3 專用：門邊熊 2m 外偵測不穩、會跟丟，不能全程視覺追蹤。置中時(熊還穩定可見)
        讀 AMCL 位姿，算出正前方 d = bear_dist - landmark_stop_distance 公尺的目標點，
        交給 _creep_to_point 閉環直行 —— 全程靠 AMCL，不依賴 YOLO，跟丟也照樣開到位。
        拿不到 AMCL 時退回計時盲推。"""
        d = max(0.0, bear_dist - self.landmark_stop_distance)
        try:
            pose, quat = self.data_processor.get_processed_amcl_pose()
        except Exception:
            pose, quat = None, None
        if pose is None or quat is None:
            print(f"[Task3] 無 AMCL → 改盲推前進 {self.landmark_blind_time:.1f}s")
            self._timed_action("FORWARD_SLOW", self.landmark_blind_time, stop_event)
            return
        yaw = math.radians(get_yaw_from_quaternion(quat[2], quat[3]))
        target = [pose[0] + d * math.cos(yaw), pose[1] + d * math.sin(yaw)]
        print(f"[Task3] 沿航向直行 {d:.2f}m → ({target[0]:.2f}, {target[1]:.2f})")
        self._creep_to_point(
            target, stop_event, tol=0.15, timeout=self.landmark_drive_timeout
        )

    def _creep_to_depth(self, target_depth, stop_event, timeout=4.0,
                        confirm_frames=1, blind_frames=3):
        """前進到 YOLO depth ≤ target_depth(或 depth 進盲區失效)才停。

        用 depth 回授把 FINE_ALIGN 那個受雜訊/網路延遲影響、會 run-to-run 飄的停車點，
        收斂到一致的近站距 → 最後的固定盲推才對得準。depth 在 ~0.4m 以下會失效(-1)，
        故 target 設在盲區邊緣即可：讀到 ≤target 或連續數幀無有效 depth(=已進盲區/熊掉到
        畫面下緣) 都視為到位。回傳此段看到的最後有效 depth(給 grasp 投影)，沒看到回 None。

        (B) confirm_frames/blind_frames：要求『連續』N 幀 ≤target 才停、『連續』M 幀無
        depth 才判進盲區 → 單幀雜訊不會提早停，停點 run-to-run 更一致。預設 1/3 = 原行為
        (Task1 不受影響)；Task3 用較嚴的值收穩停點。"""
        dp = self.data_processor
        car = self.car_controller
        print(f"[Task1] depth 閉環收斂站距 → ≤{target_depth:.2f}m (confirm={confirm_frames}, blind={blind_frames})")
        t0 = time.time()
        last_d = None
        misses = 0
        hits = 0
        while not stop_event.is_set() and (time.time() - t0) < timeout:
            info = dp.get_yolo_target_info()
            found = info is not None and info[0] == 1.0
            distance = info[1] if info is not None else 0.0
            if found and distance > 0.0:
                misses = 0
                last_d = distance
                if distance <= target_depth:
                    hits += 1
                    if hits >= confirm_frames:
                        break                   # 連續確認到達目標站距
                    car.update_action("FORWARD_SLOW")
                else:
                    hits = 0
                    car.update_action("FORWARD_SLOW")
            else:
                hits = 0
                misses += 1
                if misses >= blind_frames:       # 連續無有效 depth = 已進盲區近點 → 停
                    break
                car.update_action("FORWARD_SLOW")  # 單幀抖動：很近了，續推
            time.sleep(0.05)
        car.update_action("STOP")
        return last_d

    def _creep_forward_dist(self, dist, stop_event, timeout=6.0, fallback_time=1.0):
        """(A) 用 AMCL 位移盲推固定『距離』前進，取代固定『時間』——
        消掉馬達 ramp/摩擦/指令延遲造成的落點變異。記下起點 xy，FORWARD_SLOW 推到
        位移 ≥ dist 才停。拿不到 AMCL → fallback 計時盲推 fallback_time 秒。全程印 live 位移。"""
        car = self.car_controller
        start = self._current_xy()
        if start is None:
            print(f"[Task3] 無 AMCL → 計時盲推 {fallback_time:.1f}s")
            self._timed_action("FORWARD_SLOW", fallback_time, stop_event)
            return
        print(f"[Task3] AMCL 位移盲推 → {dist:.3f}m")
        t0 = time.time()
        last_print = 0.0
        while not stop_event.is_set() and (time.time() - t0) < timeout:
            cur = self._current_xy()
            now = time.time()
            if cur is not None:
                moved = math.hypot(cur[0] - start[0], cur[1] - start[1])
                if now - last_print >= 0.3:
                    print(f"[Task3][creep] 已前進={moved:.3f}m / {dist:.3f}m")
                    last_print = now
                if moved >= dist:
                    break
            car.update_action("FORWARD_SLOW")
            time.sleep(0.05)
        car.update_action("STOP")

    def _arm_swing_loop(self, x, z_lo, z_hi, swing_stop):
        """背景擺動：手臂在 z_lo(壓下)↔z_hi(略抬) 間反覆，前推時用一段 z 範圍刮過把手，
        補 FINE_ALIGN 的微小誤差。只動 shoulder/elbow(不碰夾爪)；由 PUSH 開執行緒呼叫，
        swing_stop.set() 後自然收尾(主執行緒 join 完才 reset_arm，無競爭)。"""
        arm = self.arm_controller
        zs = [z_lo, z_hi]
        i = 0
        while not swing_stop.is_set():
            arm.arm_to_xz(x, zs[i % 2], label="擺動")
            i += 1

    def _orient_to_yaw(self, target_yaw, stop_event, tol=8.0, timeout=12.0):
        """用 AMCL 回授原地旋轉，把車頭轉到 target_yaw (度, map frame)。

        Task1 收尾用：RELEASE 後車頭指著起點(貼牆)方向，轉回 spawn 朝向(start_yaw)
        讓車停回 home 姿態，後續任務「從起點直行」才出得去。Task2/3 朝特定方向起步也可重用。
        err 收斂到 ±180；err>0 需增加 yaw → 逆時針(與 _creep_to_point 角度慣例一致)。
        timeout 為安全上限，避免 AMCL 抖動時無限轉。"""
        car = self.car_controller
        print(f"[Nav] 回正車頭 → yaw={target_yaw:.1f}° tol={tol:.1f}")
        t0 = time.time()
        while not stop_event.is_set() and (time.time() - t0) < timeout:
            try:
                pose, quat = self.data_processor.get_processed_amcl_pose()
            except Exception:
                break
            if quat is None:
                break
            cur = get_yaw_from_quaternion(quat[2], quat[3])
            err = (target_yaw - cur + 180.0) % 360.0 - 180.0
            if abs(err) <= tol:
                break
            # 兩段速：誤差大用全速快轉(180° 才轉得完)，接近目標換慢轉收尾不過衝
            ccw = err > 0
            if abs(err) > 30.0:
                action = "COUNTERCLOCKWISE_ROTATION" if ccw else "CLOCKWISE_ROTATION"
            else:
                action = "COUNTERCLOCKWISE_ROTATION_SLOW" if ccw else "CLOCKWISE_ROTATION_SLOW"
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
