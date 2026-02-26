"""
chainablepedal_dpg_numpy_params.py
DearPyGui + Numba DSP + Audio Device Select + SR/BlockSize変更
+ エフェクトON/OFF + 削除 + プリセット保存/読込 + CPUメーター
+ パラメータ NumPy 化（DSP 最適化 & ほぼ lock-free）
現代 DAW プラグイン風チェイン型エフェクター（モノラル処理）
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
#  基本設定（GUI から変更可能）
# ============================================================

SAMPLE_RATE = 48000
BLOCK_SIZE = 256

current_input_device = None
current_output_device = None
audio_stream = None

cpu_load = 0.0
cpu_lock = threading.Lock()

PRESET_FILE = "chainable_pedal_preset.json"


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
#  エフェクトレジストリ（メタデータ）
# ============================================================

# name: (min, max, default)
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
            ("depth", 0.1, 10.0, 3.0),  # ms
            ("mix",   0.0, 1.0,  0.4),
        ],
    },
    "Delay": {
        "color": (180, 120, 255),
        "params": [
            ("time",     20.0, 800.0, 380.0),  # ms
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
    meta = EFFECT_REGISTRY[effect_type]
    return meta["params"]  # list of (name, min, max, default)


# ============================================================
#  エフェクトインスタンス（NumPy パラメータ）
# ============================================================

@dataclass
class EffectInstance:
    effect_type: str
    enabled: bool = True
    params: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))


def create_effect_instance(effect_type: str) -> EffectInstance:
    param_meta = get_param_meta(effect_type)
    defaults = [p[3] for p in param_meta]
    params = np.array(defaults, dtype=np.float32)
    return EffectInstance(effect_type=effect_type, enabled=True, params=params)


# ============================================================
#  エフェクトチェイン（Numba DSP 状態付き）
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


chain = EffectChain()
chain_lock = threading.Lock()
chain_gui_ids: List[int] = []
current_selected_index: int = 0


# ============================================================
#  DSP 状態再構築（SR変更時）
# ============================================================

def rebuild_dsp_state():
    with chain_lock:
        chain.delay_buf = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        chain.delay_write_pos = 0
        chain.reverb_buf = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
        chain.reverb_idx = 0
        chain.chorus_phase = 0.0


# ============================================================
#  オーディオコールバック
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


# ============================================================
#  オーディオデバイス / SR / BlockSize 管理
# ============================================================

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
#  DAW 風ノブ
# ============================================================

def daw_knob(label, value_getter, value_setter, min_v, max_v, width=90):
    knob_id = dpg.generate_uuid()

    with dpg.group():
        dpg.add_text(label)
        with dpg.drawlist(width=width, height=width, tag=knob_id):
            pass

    def redraw():
        dpg.delete_item(knob_id, children_only=True)
        v = value_getter()
        t = (v - min_v) / (max_v - min_v + 1e-9)
        t = max(0.0, min(1.0, t))
        angle = math.radians(225 * t - 135)

        cx, cy = width / 2, width / 2
        r = width / 2 - 8

        dpg.draw_circle((cx, cy), r, color=(60, 60, 60), thickness=4)

        color = (
            int(100 + 155 * t),
            int(80 + 50 * t),
            int(255 - 200 * t),
        )

        dpg.draw_line(
            (cx, cy),
            (cx + r * 0.75 * math.cos(angle),
             cy + r * 0.75 * math.sin(angle)),
            color=color,
            thickness=4,
        )

        dpg.draw_text((cx - 14, cy + 20), f"{v:.2f}", color=(220, 220, 220))

    redraw()

    def drag(sender, app_data):
        delta = app_data
        v = value_getter()
        v += delta * (max_v - min_v) * 0.005
        if v < min_v:
            v = min_v
        if v > max_v:
            v = max_v
        value_setter(v)
        redraw()

    dpg.set_item_drag_callback(knob_id, drag)


# ============================================================
#  チェイン GUI
# ============================================================

def rebuild_chain_gui():
    dpg.delete_item("chain_panel", children_only=True)
    global chain_gui_ids

    chain_gui_ids = []
    with chain_lock:
        effects = list(chain.effects)

    for idx, eff in enumerate(effects):
        card_id = dpg.generate_uuid()
        chain_gui_ids.append(card_id)

        color = EFFECT_REGISTRY[eff.effect_type]["color"]

        with dpg.child_window(parent="chain_panel",
                              height=70,
                              border=False,
                              tag=card_id):
            with dpg.group(horizontal=True):
                dpg.add_button(width=6, height=50,
                               enabled=False,
                               background_color=color)

                with dpg.group():
                    dpg.add_text(eff.effect_type)
                    dpg.add_text(f"#{idx}", color=(150, 150, 150))

                def make_toggle_closure(i):
                    def _toggle(sender, app_data):
                        with chain_lock:
                            chain.effects[i].enabled = app_data
                        rebuild_param_panel()
                    return _toggle

                dpg.add_checkbox(label="On",
                                 default_value=eff.enabled,
                                 callback=make_toggle_closure(idx))

                def make_delete_closure(i):
                    def _delete():
                        with chain_lock:
                            if 0 <= i < len(chain.effects):
                                chain.effects.pop(i)
                        rebuild_chain_gui()
                        rebuild_param_panel()
                    return _delete

                dpg.add_button(label="X", width=20, callback=make_delete_closure(idx))

            dpg.add_drag_payload(parent=card_id,
                                 payload_type="EFFECT_INDEX",
                                 drag_data=idx)
            dpg.add_drop_callback(card_id, drop_callback)

            def make_select_closure(i):
                def _select():
                    global current_selected_index
                    current_selected_index = i
                    rebuild_param_panel()
                return _select

            dpg.set_item_callback(card_id, make_select_closure(idx))


def drop_callback(sender, app_data):
    src_index = app_data
    with chain_lock:
        if sender not in chain_gui_ids:
            return
        dst_index = chain_gui_ids.index(sender)
        if src_index == dst_index:
            return
        eff = chain.effects.pop(src_index)
        chain.effects.insert(dst_index, eff)

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
        meta = get_param_meta(eff.effect_type)
        params_ref = eff.params

    dpg.add_text(f"{eff.effect_type} Parameters", parent="param_panel")
    dpg.add_separator(parent="param_panel")

    with dpg.group(horizontal=True, parent="param_panel"):
        for i, (name, mn, mx, df) in enumerate(meta):

            def make_getter(params_arr, index):
                return lambda: float(params_arr[index])

            def make_setter(params_arr, index):
                def _set(v):
                    params_arr[index] = v
                return _set

            daw_knob(
                label=name,
                value_getter=make_getter(params_ref, i),
                value_setter=make_setter(params_ref, i),
                min_v=mn,
                max_v=mx,
                width=90,
            )


# ============================================================
#  CPU メーター更新
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
#  メイン GUI
# ============================================================

def build_gui():
    global current_input_device, current_output_device

    dpg.create_context()
    dpg.create_viewport(title="Chainable Pedal (DearPyGui + Numba + NumPy Params)",
                        width=1150,
                        height=720)

    devices = sd.query_devices()
    device_names = [d["name"] for d in devices]

    current_input_device = sd.default.device[0]
    current_output_device = sd.default.device[1]

    with dpg.window(label="Chainable Pedal",
                    width=1150,
                    height=720,
                    no_move=True,
                    no_resize=True,
                    no_collapse=True):

        with dpg.group(horizontal=True):
            # 左パネル
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

            # 右パネル
            with dpg.child_window(width=830, border=False):
                dpg.add_text("Parameters")
                dpg.add_separator()
                with dpg.child_window(tag="param_panel",
                                      border=False,
                                      autosize_x=True,
                                      autosize_y=True):
                    pass

        with dpg.group(horizontal=True):
            dpg.add_spacer(width=10)
            dpg.add_text("Add Effect:")
            for eff_name in EFFECT_REGISTRY.keys():

                def make_add_closure(name):
                    def _add():
                        with chain_lock:
                            chain.effects.append(create_effect_instance(name))
                        rebuild_chain_gui()
                        rebuild_param_panel()
                    return _add

                dpg.add_button(label=eff_name,
                               callback=make_add_closure(eff_name))

    with chain_lock:
        chain.effects = [
            create_effect_instance("Distortion"),
            create_effect_instance("Chorus"),
            create_effect_instance("Delay"),
            create_effect_instance("Reverb"),
        ]

    rebuild_chain_gui()
    rebuild_param_panel()

    dpg.setup_dearpygui()
    dpg.show_viewport()

    with dpg.handler_registry():
        dpg.add_timer(callback=lambda: update_cpu_meter(), delay=0.2)


# ============================================================
#  メイン
# ============================================================

def warmup_numba():
    dummy = np.zeros(BLOCK_SIZE, dtype=np.float32)
    _ = dsp_distortion(dummy, np.array([5.0, 0.7], dtype=np.float32))
    _ = dsp_chorus(dummy,
                   np.array([1.0, 3.0, 0.5], dtype=np.float32),
                   SAMPLE_RATE, 0.0)
    _ = dsp_delay(dummy,
                  np.array([300.0, 0.4, 0.5], dtype=np.float32),
                  SAMPLE_RATE,
                  np.zeros(SAMPLE_RATE * 2, dtype=np.float32), 0)
    _ = dsp_reverb(dummy,
                   np.array([0.6, 0.3, 0.5], dtype=np.float32),
                   np.zeros(SAMPLE_RATE * 3, dtype=np.float32), 0)


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
    
    
