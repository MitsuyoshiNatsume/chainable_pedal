"""
chainablepedalautoload_presets_01.py
DearPyGui v2.2 対応版
NumPy パラメータ化 + Numba DSP + SR/BlockSize変更
+ エフェクトON/OFF + 削除 + プリセット保存/読込 + CPUメーター
+ FabFilter 風ノブ
"""

import math
import threading
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import sounddevice as sd
import dearpygui.dearpygui as dpg
import numba as nb


# ============================================================
#  基本設定
# ============================================================

SAMPLE_RATE = 48000
BLOCK_SIZE = 256

current_input_device = None
current_output_device = None
audio_stream = None

cpu_load = 0.0
cpu_lock = threading.Lock()

PRESET_FILE = "chainable_pedal_preset.json"

dpg.create_context()
print(dpg.get_app_configuration()["version"])


# ============================================================
#  Numba DSP 実装
# ============================================================

@nb.njit(cache=True)
def dsp_distortion(x, params):
    gain = params[0]
    mix = params[1]
    y = np.empty_like(x)
    for i in range(x.shape[0]):
        v = x[i]
        d = math.tanh(gain * v)
        y[i] = (1.0 - mix) * v + mix * d
    return y


@nb.njit(cache=True)
def dsp_chorus(x, params, sr, phase_in):
    rate = params[0]
    depth_ms = params[1]
    mix = params[2]

    n = x.shape[0]
    y = np.empty_like(x)

    max_delay_samp = int(depth_ms * sr * 0.001) + 2
    if max_delay_samp < 2:
        max_delay_samp = 2

    buf = np.zeros(max_delay_samp + n, dtype=x.dtype)

    phase = phase_in
    two_pi = 2.0 * math.pi
    for i in range(n):
        lfo = math.sin(phase)
        delay_samp = int(depth_ms * 0.5 * (1.0 + lfo) * sr * 0.001)
        if delay_samp < 0:
            delay_samp = 0
        if delay_samp > max_delay_samp - 1:
            delay_samp = max_delay_samp - 1

        buf[max_delay_samp + i] = x[i]
        d = buf[max_delay_samp + i - delay_samp]
        y[i] = (1.0 - mix) * x[i] + mix * d

        phase += two_pi * rate / sr
        if phase > two_pi:
            phase -= two_pi

    return y, phase


@nb.njit(cache=True)
def dsp_delay(x, params, sr, buf, write_pos_in):
    time_ms = params[0]
    feedback = params[1]
    mix = params[2]

    n = x.shape[0]
    y = np.empty_like(x)
    delay_samp = int(time_ms * sr * 0.001)
    if delay_samp < 1:
        delay_samp = 1
    if delay_samp >= buf.shape[0]:
        delay_samp = buf.shape[0] - 1

    write_pos = write_pos_in
    for i in range(n):
        read_pos = write_pos - delay_samp
        if read_pos < 0:
            read_pos += buf.shape[0]
        d = buf[read_pos]
        v = x[i] + feedback * d
        buf[write_pos] = v
        write_pos += 1
        if write_pos >= buf.shape[0]:
            write_pos = 0
        y[i] = (1.0 - mix) * x[i] + mix * d

    return y, write_pos


@nb.njit(cache=True)
def dsp_reverb(x, params, buf, idx_in):
    size = params[0]
    damp = params[1]
    mix = params[2]

    n = x.shape[0]
    y = np.empty_like(x)
    fb = 0.6 * size
    lp = 0.2 + 0.7 * (1.0 - damp)

    idx = idx_in
    prev = 0.0
    for i in range(n):
        v = x[i] + fb * buf[idx]
        prev = lp * v + (1.0 - lp) * prev
        buf[idx] = prev
        idx += 1
        if idx >= buf.shape[0]:
            idx = 0
        y[i] = (1.0 - mix) * x[i] + mix * prev

    return y, idx


# ============================================================
#  エフェクトレジストリ
# ============================================================

EFFECT_REGISTRY: Dict[str, Dict] = {
    "Distortion": {
        "color": (255, 140, 0),
        "params": [
            ("gain", 0.0, 20.0, 5.0),
            ("mix",  0.0, 1.0,  0.7),
        ],
    },
    "Chorus": {
        "color": (80, 160, 255),
        "params": [
            ("rate",  0.1, 5.0, 1.2),
            ("depth", 0.1, 10.0, 3.0),
            ("mix",   0.0, 1.0,  0.4),
        ],
    },
    "Delay": {
        "color": (180, 120, 255),
        "params": [
            ("time",     20.0, 800.0, 380.0),
            ("feedback", 0.0,  0.95,  0.4),
            ("mix",      0.0,  1.0,   0.35),
        ],
    },
    "Reverb": {
        "color": (120, 220, 160),
        "params": [
            ("size", 0.1, 1.0, 0.6),
            ("damp", 0.0, 1.0, 0.3),
            ("mix",  0.0, 1.0, 0.25),
        ],
    },
}


def get_param_meta(effect_type: str):
    return EFFECT_REGISTRY[effect_type]["params"]


# ============================================================
#  エフェクトインスタンス
# ============================================================

@dataclass
class EffectInstance:
    effect_type: str
    enabled: bool = True
    params: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))


def create_effect_instance(effect_type: str) -> EffectInstance:
    meta = get_param_meta(effect_type)
    defaults = [p[3] for p in meta]
    params = np.array(defaults, dtype=np.float32)
    return EffectInstance(effect_type=effect_type, enabled=True, params=params)


# ============================================================
#  エフェクトチェイン
# ============================================================

class EffectChain:
    def __init__(self):
        self.effects: List[EffectInstance] = []

        self.chorus_phase = 0.0
        self.delay_buf = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        self.delay_write_pos = 0
        self.reverb_buf = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
        self.reverb_idx = 0

    def process_block(self, x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float32)

        for eff in self.effects:
            if not eff.enabled:
                continue

            if eff.effect_type == "Distortion":
                y = dsp_distortion(y, eff.params)

            elif eff.effect_type == "Chorus":
                y, self.chorus_phase = dsp_chorus(
                    y, eff.params, SAMPLE_RATE, self.chorus_phase
                )

            elif eff.effect_type == "Delay":
                y, self.delay_write_pos = dsp_delay(
                    y, eff.params, SAMPLE_RATE,
                    self.delay_buf, self.delay_write_pos
                )

            elif eff.effect_type == "Reverb":
                y, self.reverb_idx = dsp_reverb(
                    y, eff.params, self.reverb_buf, self.reverb_idx
                )

        return y


# ============================================================
#  グローバルチェイン
# ============================================================

chain = EffectChain()
chain_lock = threading.Lock()


# ============================================================
#  FabFilter-style Knob
# ============================================================

def daw_knob(label, value_getter, value_setter, min_v, max_v, width=80, height=80):
    knob_id = dpg.generate_uuid()
    radius = width * 0.38
    ring_width = 6

    state = {"dragging": False, "last_y": 0.0}

    def redraw():
        dpg.delete_item(knob_id, children_only=True)

        v = value_getter()
        t = (v - min_v) / (max_v - min_v)
        t = max(0.0, min(1.0, t))

        angle = -135 + t * 270
        rad = math.radians(angle)

        cx = width * 0.5
        cy = height * 0.5

        # 外周リング
        dpg.draw_circle((cx, cy), radius, color=(80, 80, 80),
                        thickness=ring_width, parent=knob_id)

        # 針
        x2 = cx + radius * math.cos(rad)
        y2 = cy + radius * math.sin(rad)
        dpg.draw_line((cx, cy), (x2, y2),
                      color=(255, 255, 255),
                      thickness=3,
                      parent=knob_id)

        # 針の根本の小さな丸
        dpg.draw_circle((cx, cy), 4, color=(255, 255, 255),
                        fill=(255, 255, 255), parent=knob_id)

    def on_mouse_down(sender, app_data):
        state["dragging"] = True
        _, y = dpg.get_mouse_pos()
        state["last_y"] = y

    def on_mouse_drag(sender, app_data):
        if not state["dragging"]:
            return

        _, y = dpg.get_mouse_pos()
        dy = state["last_y"] - y
        state["last_y"] = y

        delta = dy * (max_v - min_v) * 0.002
        v = value_getter() + delta
        v = max(min_v, min(max_v, v))
        value_setter(v)
        redraw()

        if not dpg.is_item_active(knob_id):
            state["dragging"] = False

    with dpg.group():
        dpg.add_text(label)
        dpg.add_drawlist(width=width, height=height, tag=knob_id)

        with dpg.item_handler_registry() as handler:
            dpg.add_item_clicked_handler(callback=on_mouse_down)
            dpg.add_item_active_handler(callback=on_mouse_drag)

        dpg.bind_item_handler_registry(knob_id, handler)

    redraw()


# ============================================================
#  テーマ
# ============================================================

def create_color_theme(color):
    with dpg.theme() as theme_id:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, color)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, color)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, color)
    return theme_id


def add_color_bar(parent, color, width=6, height=50):
    btn_id = dpg.add_button(parent=parent, width=width, height=height, enabled=False)
    theme_id = create_color_theme(color)
    dpg.bind_item_theme(btn_id, theme_id)
    return btn_id


# ============================================================
#  チェイン GUI
# ============================================================

chain_gui_ids: List[int] = []
current_selected_index: int = 0


def drop_callback(sender, app_data):
    # app_data は drag_data で渡した int
    src_index = int(app_data)

    with chain_lock:
        if sender not in chain_gui_ids:
            return

        dst_index = chain_gui_ids.index(sender)

        if src_index == dst_index:
            return
        if not (0 <= src_index < len(chain.effects)):
            return

        eff = chain.effects.pop(src_index)
        chain.effects.insert(dst_index, eff)

    rebuild_chain_gui()
    rebuild_param_panel()




def rebuild_chain_gui():
    global chain_gui_ids, current_selected_index

    dpg.delete_item("chain_panel", children_only=True)
    chain_gui_ids = []

    with chain_lock:
        effects = list(chain.effects)
        n = len(effects)

    for idx, eff in enumerate(effects):
        eff_name = eff.effect_type
        color = EFFECT_REGISTRY[eff_name]["color"]

        # カードのタグ
        card_tag = f"effect_card_{idx}"
        chain_gui_ids.append(card_tag)

        # カード本体
        dpg.add_selectable(
            label=f"{idx+1}. {eff_name}",
            tag=card_tag,
            span_columns=True,
            callback=lambda s, a: on_select_effect(s),
            parent="chain_panel"
        )

        # 選択状態の復元
        if idx == current_selected_index:
            dpg.set_value(card_tag, True)

        # 上下移動ボタン
        with dpg.group(horizontal=True, parent="chain_panel"):
            up_tag = f"effect_up_{idx}"
            down_tag = f"effect_down_{idx}"
            remove_tag = f"effect_remove_{idx}"

            dpg.add_button(label="UP", tag=up_tag,
                           callback=lambda s, a: on_move_effect(s))

            dpg.add_button(label="DOWN", tag=down_tag,
                           callback=lambda s, a: on_move_effect(s))

            dpg.add_button(label="REMOVE", tag=remove_tag,
                           callback=lambda s, a: on_remove_effect(s))

        dpg.add_separator(parent="chain_panel")


def on_select_effect(sender):
    global current_selected_index

    idx = int(sender.split("_")[-1])
    current_selected_index = idx

    # ハイライト更新
    for i, tag in enumerate(chain_gui_ids):
        dpg.set_value(tag, i == idx)

    rebuild_param_panel()


def on_move_effect(sender):
    global current_selected_index

    parts = sender.split("_")
    direction = parts[-2]   # "up" or "down"
    idx = int(parts[-1])

    with chain_lock:
        if direction == "up" and idx > 0:
            chain.effects[idx], chain.effects[idx - 1] = chain.effects[idx - 1], chain.effects[idx]
            current_selected_index = idx - 1

        elif direction == "down" and idx < len(chain.effects) - 1:
            chain.effects[idx], chain.effects[idx + 1] = chain.effects[idx + 1], chain.effects[idx]
            current_selected_index = idx + 1

    rebuild_chain_gui()
    rebuild_param_panel()


def on_remove_effect(sender):
    global current_selected_index

    idx = int(sender.split("_")[-1])

    with chain_lock:
        if 0 <= idx < len(chain.effects):
            chain.effects.pop(idx)

            # 選択位置調整
            if current_selected_index >= len(chain.effects):
                current_selected_index = max(0, len(chain.effects) - 1)

    rebuild_chain_gui()
    rebuild_param_panel()



# ============================================================
#  パラメータパネル
# ============================================================

def rebuild_param_panel():
    dpg.delete_item("param_panel", children_only=True)

    with chain_lock:
        if not chain.effects:
            dpg.add_text("No effects in chain.", parent="param_panel")
            return

        idx = min(current_selected_index, len(chain.effects) - 1)
        eff = chain.effects[idx]
        meta = get_param_meta(eff.effect_type)   # [(name, min, max, default), ...]
        params_ref = eff.params                  # 実際の値リスト

    dpg.add_text(f"{eff.effect_type} Parameters", parent="param_panel")
    dpg.add_separator(parent="param_panel")

    # スライダーを縦に並べる
    for i, (name, mn, mx, df) in enumerate(meta):

        # 値の getter/setter
        def make_setter(params_arr, index):
            def _set(sender, value):
                params_arr[index] = value
            return _set

        dpg.add_text(name, parent="param_panel")

        dpg.add_slider_float(
            label="",
            default_value=params_ref[i],
            min_value=mn,
            max_value=mx,
            width=300,
            callback=make_setter(params_ref, i),
            parent="param_panel"
        )

        dpg.add_spacer(height=5, parent="param_panel")

# ============================================================
#  CPU メーター
# ============================================================

def update_cpu_meter():
    with cpu_lock:
        load = cpu_load

    if load < 0:
        load = 0.0
    if load > 2.0:
        load = 2.0

    dpg.set_value("cpu_text", f"CPU Load: {load*100:.1f}%")
    dpg.set_value("cpu_bar", min(load, 1.0))


# ============================================================
#  オーディオ関連コールバック
# ============================================================

def audio_callback(indata, outdata, frames, time_info, status):
    global cpu_load
    if status:
        print(status)

    start = time.perf_counter()

    x = indata[:, 0].copy()

    with chain_lock:
        y = chain.process_block(x)

    outdata[:, 0] = y
    outdata[:, 1] = y

    end = time.perf_counter()
    block_time = frames / float(SAMPLE_RATE)

    with cpu_lock:
        cpu_load = (end - start) / block_time if block_time > 0 else 0.0


def restart_audio_stream():
    global audio_stream, current_input_device, current_output_device

    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()

    audio_stream = sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=2,
        dtype="float32",
        callback=audio_callback,
        device=(current_input_device, current_output_device),
    )
    audio_stream.start()


def on_input_device_change(sender, app_data):
    global current_input_device
    current_input_device = app_data
    restart_audio_stream()


def on_output_device_change(sender, app_data):
    global current_output_device
    current_output_device = app_data
    restart_audio_stream()


def on_sample_rate_change(sender, app_data):
    global SAMPLE_RATE
    SAMPLE_RATE = int(app_data)
    rebuild_dsp_state()
    restart_audio_stream()


def on_block_size_change(sender, app_data):
    global BLOCK_SIZE
    BLOCK_SIZE = int(app_data)
    restart_audio_stream()


# ============================================================
#  DSP 状態再構築
# ============================================================

def rebuild_dsp_state():
    with chain_lock:
        chain.delay_buf = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        chain.delay_write_pos = 0
        chain.reverb_buf = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
        chain.reverb_idx = 0
        chain.chorus_phase = 0.0


# ============================================================
#  Numba ウォームアップ
# ============================================================

def warmup_numba():
    dummy = np.zeros(BLOCK_SIZE, dtype=np.float32)

    _ = dsp_distortion(dummy, np.array([5.0, 0.7], dtype=np.float32))

    _ = dsp_chorus(
        dummy,
        np.array([1.0, 3.0, 0.5], dtype=np.float32),
        SAMPLE_RATE,
        0.0
    )

    _ = dsp_delay(
        dummy,
        np.array([300.0, 0.4, 0.5], dtype=np.float32),
        SAMPLE_RATE,
        np.zeros(SAMPLE_RATE * 2, dtype=np.float32),
        0
    )

    _ = dsp_reverb(
        dummy,
        np.array([0.6, 0.3, 0.5], dtype=np.float32),
        np.zeros(SAMPLE_RATE * 3, dtype=np.float32),
        0
    )


# ============================================================
#  プリセット保存 / 読み込み
# ============================================================

def save_preset(sender, app_data):
    data = []

    with chain_lock:
        for eff in chain.effects:
            meta = get_param_meta(eff.effect_type)
            params_dict = {}

            for i, (name, mn, mx, df) in enumerate(meta):
                params_dict[name] = float(eff.params[i])

            data.append({
                "type": eff.effect_type,
                "enabled": eff.enabled,
                "params": params_dict,
            })

    try:
        with open(PRESET_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("Preset saved:", PRESET_FILE)
    except Exception as e:
        print("Preset save error:", e)


def load_preset(sender, app_data):
    global current_selected_index

    try:
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Preset load error:", e)
        return

    with chain_lock:
        chain.effects = []

        for eff_data in data:
            etype = eff_data.get("type", "Distortion")
            if etype not in EFFECT_REGISTRY:
                continue

            eff = create_effect_instance(etype)
            eff.enabled = eff_data.get("enabled", True)

            params_dict = eff_data.get("params", {})
            meta = get_param_meta(etype)

            for i, (name, mn, mx, df) in enumerate(meta):
                if name in params_dict:
                    v = float(params_dict[name])
                    if v < mn:
                        v = mn
                    if v > mx:
                        v = mx
                    eff.params[i] = v

            chain.effects.append(eff)

        current_selected_index = 0

    rebuild_chain_gui()
    rebuild_param_panel()


# ============================================================
#  GUI 構築
# ============================================================

def build_gui():
    global current_input_device, current_output_device

    dpg.create_context()
    dpg.create_viewport(title="Chainable Pedal (DearPyGui v2.2)",
                        width=1150,
                        height=720)

    devices = sd.query_devices()
    print(devices)
    #device_names = [d["name"] for d in devices]
    #device_names = [f"{i}: {d['name']}" for i, d in enumerate(devices)]
    device_names = [
        f"{i}: {d['name']} "
        f"(In:{d['max_input_channels']} Out:{d['max_output_channels']})"
        for i, d in enumerate(devices)
    ]
    print(device_names)

    current_input_device = sd.default.device[0]
    current_output_device = sd.default.device[1]

    with dpg.window(label="Chainable Pedal",
                    width=1150,
                    height=720,
                    no_move=True,
                    no_resize=True,
                    no_collapse=True):

        with dpg.group(horizontal=True):

            # -------------------------
            # 左パネル
            # -------------------------
            with dpg.child_window(width=300, border=False):
                dpg.add_text("Effect Chain")
                dpg.add_separator()

                with dpg.child_window(tag="chain_panel",
                                      border=False,
                                      autosize_x=True,
                                      autosize_y=True):
                    pass

                dpg.add_separator()
                dpg.add_text("Audio Devices")

                dpg.add_text("Input:")
                dpg.add_combo(
                    device_names,
                    default_value=device_names[current_input_device],
                    callback=lambda s, a: on_input_device_change(
                        s, device_names.index(a)
                    ),
                )

                dpg.add_text("Output:")
                dpg.add_combo(
                    device_names,
                    default_value=device_names[current_output_device],
                    callback=lambda s, a: on_output_device_change(
                        s, device_names.index(a)
                    ),
                )

                dpg.add_separator()
                dpg.add_text("Audio Settings")

                sample_rates = ["44100", "48000", "96000"]
                block_sizes = ["128", "256", "512", "1024"]

                dpg.add_text("Sample Rate:")
                dpg.add_combo(
                    sample_rates,
                    default_value=str(SAMPLE_RATE),
                    callback=on_sample_rate_change,
                )

                dpg.add_text("Block Size:")
                dpg.add_combo(
                    block_sizes,
                    default_value=str(BLOCK_SIZE),
                    callback=on_block_size_change,
                )

                dpg.add_separator()
                dpg.add_text("Preset")
                dpg.add_button(label="Save Preset", callback=save_preset)
                dpg.add_button(label="Load Preset", callback=load_preset)

                dpg.add_separator()
                dpg.add_text("Performance")
                dpg.add_text("CPU Load: 0.0%", tag="cpu_text")
                dpg.add_progress_bar(tag="cpu_bar", default_value=0.0, width=250)

            # -------------------------
            # 右パネル
            # -------------------------
            with dpg.child_window(width=830, border=False):
                dpg.add_text("Parameters")
                dpg.add_separator()

                with dpg.child_window(tag="param_panel",
                                      border=False,
                                      autosize_x=True,
                                      autosize_y=True):
                    pass

        # -------------------------
        # エフェクト追加ボタン
        # -------------------------
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=10)
            dpg.add_text("Add Effect:")

            for eff_name in EFFECT_REGISTRY.keys():

                def make_add_closure(name):
                    def _add(sender, app_data):
                        with chain_lock:
                            chain.effects.append(create_effect_instance(name))
                        rebuild_chain_gui()
                        rebuild_param_panel()
                    return _add

                dpg.add_button(label=eff_name,
                               callback=make_add_closure(eff_name))

    # -------------------------
    # 初期エフェクト
    # -------------------------
    with chain_lock:
        chain.effects = [
            create_effect_instance("Distortion"),
            create_effect_instance("Chorus"),
            create_effect_instance("Delay"),
            create_effect_instance("Reverb"),
        ]

    rebuild_chain_gui()
    rebuild_param_panel()

    with dpg.font_registry():
        with dpg.font("C:/Windows/Fonts/meiryo.ttc", 18) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)

    dpg.bind_font(default_font)

    dpg.setup_dearpygui()
    dpg.show_viewport()


    # -------------------------
    # CPU メーター更新（set_frame_callback を使用）
    # -------------------------
    last_cpu_update = {"time": 0}

    def cpu_meter_update():
        import time
        now = time.time()
        if now - last_cpu_update["time"] >= 0.2:
            update_cpu_meter()
            last_cpu_update["time"] = now

        # 次のフレームで再度呼ぶ
        dpg.set_frame_callback(1, cpu_meter_update)

    # 最初の呼び出し
    dpg.set_frame_callback(1, cpu_meter_update)








# ============================================================
#  メイン
# ============================================================

def main():
    warmup_numba()
    build_gui()
    rebuild_dsp_state()
    restart_audio_stream()

    try:
        dpg.start_dearpygui()
    finally:
        if audio_stream:
            audio_stream.stop()
            audio_stream.close()
        dpg.destroy_context()


if __name__ == "__main__":
    main()