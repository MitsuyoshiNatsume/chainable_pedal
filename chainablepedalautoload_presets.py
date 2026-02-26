# chainable_pedal_autoload_presets.py
import sys
import os
import json
import time
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets, QtCore
from numba import njit

# ====== 設定 ======
SR = 48000
BLOCKSIZE = 256
CHANNELS = 2
DTYPE = np.float32
PRESET_DIR = "presets"

os.makedirs(PRESET_DIR, exist_ok=True)

# ====== numba コア関数群 ======
@njit(cache=True)
def tanh_distort_block(x, out, gain, b0, a1, y_prev, drywet):
    y = y_prev
    for i in range(x.shape[0]):
        xd = np.tanh(gain * x[i])
        yn = b0 * xd - a1 * y
        out[i] = drywet * yn + (1.0 - drywet) * x[i]
        y = yn
    return y

@njit(cache=True)
def chorus_block(x, out, sr, depth_ms, rate_hz, mix, lfo_phase, delay_buf, buf_pos):
    N = x.shape[0]
    buf_len = delay_buf.shape[0]
    phase = lfo_phase
    two_pi = 2.0 * np.pi
    depth_samples = depth_ms * sr / 1000.0
    for i in range(N):
        lfo = np.sin(phase)
        phase += two_pi * rate_hz / sr
        if phase > two_pi:
            phase -= two_pi
        delay = 1.5 + (lfo * depth_samples)
        read_pos = buf_pos - int(delay)
        frac = delay - int(delay)
        if read_pos < 0:
            read_pos += buf_len
        read_pos_next = read_pos + 1
        if read_pos_next >= buf_len:
            read_pos_next -= buf_len
        s0 = delay_buf[read_pos]
        s1 = delay_buf[read_pos_next]
        delayed = (1.0 - frac) * s0 + frac * s1
        delay_buf[buf_pos] = x[i]
        buf_pos += 1
        if buf_pos >= buf_len:
            buf_pos = 0
        out[i] = (1.0 - mix) * x[i] + mix * delayed
    return phase, buf_pos

@njit(cache=True)
def delay_block(x, out, buf, buf_pos, buf_len, delay_samples, feedback, mix):
    N = x.shape[0]
    ypos = buf_pos
    for i in range(N):
        read_pos = ypos - int(delay_samples)
        frac = delay_samples - int(delay_samples)
        if read_pos < 0:
            read_pos += buf_len
        read_pos_next = read_pos + 1
        if read_pos_next >= buf_len:
            read_pos_next -= buf_len
        s0 = buf[read_pos]
        s1 = buf[read_pos_next]
        delayed = (1.0 - frac) * s0 + frac * s1
        out[i] = (1.0 - mix) * x[i] + mix * delayed
        buf[ypos] = x[i] + delayed * feedback
        ypos += 1
        if ypos >= buf_len:
            ypos = 0
    return ypos

@njit(cache=True)
def reverb_block(x, out, buf, buf_len, pos, taps, gains, lp_b0, lp_a1, lp_state):
    N = x.shape[0]
    for i in range(N):
        inp = x[i]
        acc = 0.0
        for t in range(taps.shape[0]):
            tap = taps[t]
            read_pos = pos - int(tap)
            if read_pos < 0:
                read_pos += buf_len
            acc += buf[read_pos] * gains[t]
        wet = acc
        out[i] = 0.5 * inp + 0.5 * wet
        fb = inp + wet * 0.7
        lp_out = lp_b0 * fb - lp_a1 * lp_state[0]
        lp_state[0] = lp_out
        buf[pos] = lp_out
        pos += 1
        if pos >= buf_len:
            pos = 0
    return pos

def one_pole_coeffs(cutoff, sr):
    x = np.exp(-2.0 * np.pi * cutoff / sr)
    b0 = 1.0 - x
    a1 = -x
    return np.float32(b0), np.float32(a1)

# ====== エフェクトクラス群 ======
class Distortion:
    def __init__(self, sr, blocksize):
        self.sr = sr; self.blocksize = blocksize
        self.gain = 6.0; self.cutoff = 6000.0; self.drywet = 1.0
        self.b0, self.a1 = one_pole_coeffs(self.cutoff, sr)
        self.y_prev = np.float32(0.0)
        self.tmp = np.zeros(blocksize, dtype=DTYPE)
        self.enabled = True; self.name = "Distortion"

    def clone(self):
        new = Distortion(self.sr, self.blocksize)
        new.gain = float(self.gain); new.cutoff = float(self.cutoff); new.drywet = float(self.drywet)
        new.b0 = float(self.b0); new.a1 = float(self.a1); new.y_prev = np.float32(self.y_prev)
        new.enabled = bool(self.enabled)
        return new

    def to_dict(self):
        return {"type":"Distortion","gain":self.gain,"cutoff":self.cutoff,"drywet":self.drywet,"enabled":self.enabled}

    @staticmethod
    def from_dict(d, sr, blocksize):
        e = Distortion(sr, blocksize)
        e.gain = float(d.get("gain", e.gain))
        e.cutoff = float(d.get("cutoff", e.cutoff))
        e.drywet = float(d.get("drywet", e.drywet))
        e.enabled = bool(d.get("enabled", e.enabled))
        return e

    def process(self, in_mono):
        if not self.enabled:
            self.tmp[:in_mono.shape[0]] = in_mono; return self.tmp
        self.b0, self.a1 = one_pole_coeffs(self.cutoff, self.sr)
        self.y_prev = tanh_distort_block(in_mono, self.tmp,
                                         np.float32(self.gain),
                                         np.float32(self.b0),
                                         np.float32(self.a1),
                                         np.float32(self.y_prev),
                                         np.float32(self.drywet))
        return self.tmp

class Chorus:
    def __init__(self, sr, blocksize):
        self.sr = sr; self.blocksize = blocksize
        self.depth_ms = 5.0; self.rate_hz = 0.8; self.mix = 0.5
        self.buf_len = int(sr * 2.0); self.buf = np.zeros(self.buf_len, dtype=DTYPE)
        self.pos = 0; self.phase = 0.0; self.tmp = np.zeros(blocksize, dtype=DTYPE)
        self.enabled = True; self.name = "Chorus"

    def clone(self):
        new = Chorus(self.sr, self.blocksize)
        new.depth_ms = float(self.depth_ms); new.rate_hz = float(self.rate_hz); new.mix = float(self.mix)
        new.pos = int(self.pos); new.phase = float(self.phase); new.buf = np.copy(self.buf)
        new.enabled = bool(self.enabled)
        return new

    def to_dict(self):
        return {"type":"Chorus","depth_ms":self.depth_ms,"rate_hz":self.rate_hz,"mix":self.mix,"enabled":self.enabled}

    @staticmethod
    def from_dict(d, sr, blocksize):
        e = Chorus(sr, blocksize)
        e.depth_ms = float(d.get("depth_ms", e.depth_ms))
        e.rate_hz = float(d.get("rate_hz", e.rate_hz))
        e.mix = float(d.get("mix", e.mix))
        e.enabled = bool(d.get("enabled", e.enabled))
        return e

    def process(self, in_mono):
        if not self.enabled:
            self.tmp[:in_mono.shape[0]] = in_mono; return self.tmp
        phase, pos = chorus_block(in_mono, self.tmp, self.sr,
                                  np.float32(self.depth_ms),
                                  np.float32(self.rate_hz),
                                  np.float32(self.mix),
                                  np.float32(self.phase),
                                  self.buf, self.pos)
        self.phase = phase; self.pos = pos
        return self.tmp

class Delay:
    def __init__(self, sr, blocksize):
        self.sr = sr; self.blocksize = blocksize
        self.time_ms = 350.0; self.feedback = 0.35; self.mix = 0.35
        self.max_sec = 5.0; self.buf_len = int(sr * self.max_sec); self.buf = np.zeros(self.buf_len, dtype=DTYPE)
        self.pos = 0; self.tmp = np.zeros(blocksize, dtype=DTYPE)
        self.enabled = True; self.name = "Delay"

    def clone(self):
        new = Delay(self.sr, self.blocksize)
        new.time_ms = float(self.time_ms); new.feedback = float(self.feedback); new.mix = float(self.mix)
        new.pos = int(self.pos); new.buf = np.copy(self.buf); new.enabled = bool(self.enabled)
        return new

    def to_dict(self):
        return {"type":"Delay","time_ms":self.time_ms,"feedback":self.feedback,"mix":self.mix,"enabled":self.enabled}

    @staticmethod
    def from_dict(d, sr, blocksize):
        e = Delay(sr, blocksize)
        e.time_ms = float(d.get("time_ms", e.time_ms))
        e.feedback = float(d.get("feedback", e.feedback))
        e.mix = float(d.get("mix", e.mix))
        e.enabled = bool(d.get("enabled", e.enabled))
        return e

    def process(self, in_mono):
        if not self.enabled:
            self.tmp[:in_mono.shape[0]] = in_mono; return self.tmp
        delay_samples = self.time_ms * self.sr / 1000.0
        self.pos = delay_block(in_mono, self.tmp, self.buf, self.pos,
                               self.buf_len, np.float32(delay_samples),
                               np.float32(self.feedback),
                               np.float32(self.mix))
        return self.tmp

class Reverb:
    def __init__(self, sr, blocksize):
        self.sr = sr; self.blocksize = blocksize
        self.taps = np.array([int(sr*0.029), int(sr*0.037), int(sr*0.043), int(sr*0.053)], dtype=np.int32)
        self.gains = np.array([0.7, 0.6, 0.5, 0.4], dtype=DTYPE)
        self.buf_len = int(sr * 6.0); self.buf = np.zeros(self.buf_len, dtype=DTYPE)
        self.pos = 0; self.lp_cutoff = 6000.0; self.lp_b0, self.lp_a1 = one_pole_coeffs(self.lp_cutoff, sr)
        self.lp_state = np.zeros(1, dtype=DTYPE); self.tmp = np.zeros(blocksize, dtype=DTYPE)
        self.enabled = True; self.mix = 0.4; self.name = "Reverb"

    def clone(self):
        new = Reverb(self.sr, self.blocksize)
        new.taps = np.copy(self.taps); new.gains = np.copy(self.gains); new.buf = np.copy(self.buf)
        new.pos = int(self.pos); new.lp_cutoff = float(self.lp_cutoff); new.lp_b0 = float(self.lp_b0)
        new.lp_a1 = float(self.lp_a1); new.lp_state = np.copy(self.lp_state); new.enabled = bool(self.enabled)
        new.mix = float(self.mix)
        return new

    def to_dict(self):
        return {"type":"Reverb","mix":self.mix,"lp_cutoff":self.lp_cutoff,"enabled":self.enabled}

    @staticmethod
    def from_dict(d, sr, blocksize):
        e = Reverb(sr, blocksize)
        e.mix = float(d.get("mix", e.mix))
        e.lp_cutoff = float(d.get("lp_cutoff", e.lp_cutoff))
        e.lp_b0, e.lp_a1 = one_pole_coeffs(e.lp_cutoff, sr)
        e.enabled = bool(d.get("enabled", e.enabled))
        return e

    def process(self, in_mono):
        if not self.enabled:
            self.tmp[:in_mono.shape[0]] = in_mono; return self.tmp
        self.pos = reverb_block(in_mono, self.tmp, self.buf, self.buf_len, self.pos,
                                self.taps, self.gains, np.float32(self.lp_b0), np.float32(self.lp_a1), self.lp_state)
        out = (1.0 - self.mix) * in_mono + self.mix * self.tmp
        self.tmp[:in_mono.shape[0]] = out
        return self.tmp

# ====== チェイン参照（copy-on-write） ======
_initial_chain = [Distortion(SR, BLOCKSIZE), Chorus(SR, BLOCKSIZE), Delay(SR, BLOCKSIZE)]
chain_ref = _initial_chain

# ====== 初回コンパイル（事前） ======
_dummy = np.zeros(BLOCKSIZE, dtype=DTYPE)
try:
    _tmp = np.zeros(BLOCKSIZE, dtype=DTYPE)
    _ = tanh_distort_block(_dummy, _tmp, np.float32(1.0), np.float32(0.5), np.float32(-0.5), np.float32(0.0), np.float32(1.0))
    _ = chorus_block(_dummy, _tmp, SR, np.float32(5.0), np.float32(0.5), np.float32(0.5), np.float32(0.0), np.zeros(1024, dtype=DTYPE), 0)
    _ = delay_block(_dummy, _tmp, np.zeros(1024, dtype=DTYPE), 0, 1024, np.float32(100.0), np.float32(0.3), np.float32(0.3))
    _ = reverb_block(_dummy, _tmp, np.zeros(2048, dtype=DTYPE), 2048, 0, np.array([10,20], dtype=np.int32), np.array([0.5,0.3], dtype=DTYPE), np.float32(0.5), np.float32(-0.5), np.zeros(1, dtype=DTYPE))
except Exception:
    pass

# ====== プリセット保存 / 読み込みユーティリティ ======
def chain_to_preset_dict(chain):
    preset = []
    for eff in chain:
        if hasattr(eff, "to_dict"):
            preset.append(eff.to_dict())
    return {"chain": preset, "timestamp": time.time()}

def preset_dict_to_chain(d):
    new_chain = []
    for item in d.get("chain", []):
        t = item.get("type", "")
        if t == "Distortion":
            new_chain.append(Distortion.from_dict(item, SR, BLOCKSIZE))
        elif t == "Chorus":
            new_chain.append(Chorus.from_dict(item, SR, BLOCKSIZE))
        elif t == "Delay":
            new_chain.append(Delay.from_dict(item, SR, BLOCKSIZE))
        elif t == "Reverb":
            new_chain.append(Reverb.from_dict(item, SR, BLOCKSIZE))
    return new_chain

def save_preset_file(name, chain):
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).rstrip()
    path = os.path.join(PRESET_DIR, f"{safe_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chain_to_preset_dict(chain), f, ensure_ascii=False, indent=2)
    return path

def load_preset_file(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return preset_dict_to_chain(d)

def list_presets():
    files = []
    for fn in os.listdir(PRESET_DIR):
        if fn.lower().endswith(".json"):
            files.append(fn[:-5])
    files.sort()
    return files

def most_recent_preset_path():
    files = [os.path.join(PRESET_DIR, fn) for fn in os.listdir(PRESET_DIR) if fn.lower().endswith(".json")]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

# ====== オーディオコールバック ======
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print("Stream status:", status)
    x = indata[:frames, 0].astype(DTYPE, copy=False)
    local_chain = chain_ref
    work = x.copy()
    for eff in local_chain:
        work = eff.process(work)
    if outdata.shape[1] >= 2:
        outdata[:frames, 0] = work
        outdata[:frames, 1] = work
    else:
        outdata[:frames, 0] = work

# ====== GUI：チェイン編集とプリセットUI（自動ロード対応） ======
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chainable Pedal with Auto-Load Presets")
        self.resize(900, 520)
        layout = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()

        # Chain list
        self.list_widget = QtWidgets.QListWidget()
        self.refresh_chain_list()
        left.addWidget(QtWidgets.QLabel("Effect Chain"))
        left.addWidget(self.list_widget)

        # Buttons for chain
        btn_layout = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add")
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.up_btn = QtWidgets.QPushButton("Up")
        self.down_btn = QtWidgets.QPushButton("Down")
        self.bypass_btn = QtWidgets.QPushButton("Toggle Bypass")
        btn_layout.addWidget(self.add_btn); btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.up_btn); btn_layout.addWidget(self.down_btn)
        btn_layout.addWidget(self.bypass_btn)
        left.addLayout(btn_layout)

        self.add_btn.clicked.connect(self.on_add)
        self.remove_btn.clicked.connect(self.on_remove)
        self.up_btn.clicked.connect(self.on_up)
        self.down_btn.clicked.connect(self.on_down)
        self.bypass_btn.clicked.connect(self.on_bypass)
        self.list_widget.currentRowChanged.connect(self.on_select)

        # Parameter area
        self.param_area = QtWidgets.QGroupBox("Parameters")
        p_layout = QtWidgets.QFormLayout()
        self.param_area.setLayout(p_layout)
        right.addWidget(self.param_area)

        # Preset controls
        preset_layout = QtWidgets.QHBoxLayout()
        self.preset_combo = QtWidgets.QComboBox()
        self.refresh_preset_list()
        self.load_preset_btn = QtWidgets.QPushButton("Load Preset")
        self.save_preset_btn = QtWidgets.QPushButton("Save Preset")
        self.delete_preset_btn = QtWidgets.QPushButton("Delete Preset")
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.load_preset_btn)
        preset_layout.addWidget(self.save_preset_btn)
        preset_layout.addWidget(self.delete_preset_btn)
        right.addLayout(preset_layout)

        self.load_preset_btn.clicked.connect(self.on_load_preset)
        self.save_preset_btn.clicked.connect(self.on_save_preset)
        self.delete_preset_btn.clicked.connect(self.on_delete_preset)

        # Device info and stop
        self.device_label = QtWidgets.QLabel("Devices: see console")
        right.addWidget(self.device_label)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(QtWidgets.QApplication.quit)
        right.addWidget(self.stop_btn)

        layout.addLayout(left, 2)
        layout.addLayout(right, 3)
        self.setLayout(layout)

        self.param_widgets = []
        if len(chain_ref) > 0:
            self.list_widget.setCurrentRow(0)

    # chain UI helpers
    def refresh_chain_list(self):
        self.list_widget.clear()
        for eff in chain_ref:
            state = "[On]" if eff.enabled else "[Bypass]"
            self.list_widget.addItem(f"{state} {eff.name}")

    def on_add(self):
        items = ("Distortion", "Chorus", "Delay", "Reverb")
        item, ok = QtWidgets.QInputDialog.getItem(self, "Add Effect", "Type:", items, 0, False)
        if not ok:
            return
        if item == "Distortion":
            new = Distortion(SR, BLOCKSIZE)
        elif item == "Chorus":
            new = Chorus(SR, BLOCKSIZE)
        elif item == "Delay":
            new = Delay(SR, BLOCKSIZE)
        else:
            new = Reverb(SR, BLOCKSIZE)
        new_chain = list(chain_ref); new_chain.append(new)
        self.swap_chain(new_chain); self.refresh_chain_list()
        self.list_widget.setCurrentRow(len(new_chain)-1)

    def on_remove(self):
        idx = self.list_widget.currentRow()
        if idx < 0: return
        new_chain = list(chain_ref); new_chain.pop(idx)
        self.swap_chain(new_chain); self.refresh_chain_list()
        self.list_widget.setCurrentRow(max(0, idx-1))

    def on_up(self):
        idx = self.list_widget.currentRow()
        if idx <= 0: return
        new_chain = list(chain_ref); new_chain[idx-1], new_chain[idx] = new_chain[idx], new_chain[idx-1]
        self.swap_chain(new_chain); self.refresh_chain_list(); self.list_widget.setCurrentRow(idx-1)

    def on_down(self):
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(chain_ref)-1: return
        new_chain = list(chain_ref); new_chain[idx+1], new_chain[idx] = new_chain[idx], new_chain[idx+1]
        self.swap_chain(new_chain); self.refresh_chain_list(); self.list_widget.setCurrentRow(idx+1)

    def on_bypass(self):
        idx = self.list_widget.currentRow()
        if idx < 0: return
        new_chain = list(chain_ref)
        new_eff = new_chain[idx].clone(); new_eff.enabled = not new_eff.enabled
        new_chain[idx] = new_eff; self.swap_chain(new_chain); self.refresh_chain_list(); self.list_widget.setCurrentRow(idx)

    def clear_param_area(self):
        layout = self.param_area.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()
        self.param_widgets = []

    def build_param_ui_for(self, eff, idx):
        self.clear_param_area()
        layout = self.param_area.layout()

        if isinstance(eff, Distortion):
            gain_label = QtWidgets.QLabel(f"Gain: {eff.gain:.2f}")
            gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); gain_slider.setRange(0, 1200); gain_slider.setValue(int(eff.gain * 100))
            def on_gain_change(v):
                gain_label.setText(f"Gain: {v/100.0:.2f}")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.gain = v / 100.0; new_chain[idx] = new_eff; self.swap_chain(new_chain)
            gain_slider.valueChanged.connect(on_gain_change)

            cut_label = QtWidgets.QLabel(f"Cutoff: {eff.cutoff:.0f} Hz")
            cut_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); cut_slider.setRange(20, 20000); cut_slider.setValue(int(eff.cutoff))
            def on_cut_change(v):
                cut_label.setText(f"Cutoff: {v} Hz")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.cutoff = float(v); new_chain[idx] = new_eff; self.swap_chain(new_chain)
            cut_slider.valueChanged.connect(on_cut_change)

            dw_label = QtWidgets.QLabel(f"Dry/Wet: {eff.drywet:.2f}")
            dw_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); dw_slider.setRange(0, 1000); dw_slider.setValue(int(eff.drywet * 1000))
            def on_dw_change(v):
                dw_label.setText(f"Dry/Wet: {v/1000.0:.2f}")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.drywet = v / 1000.0; new_chain[idx] = new_eff; self.swap_chain(new_chain)
            dw_slider.valueChanged.connect(on_dw_change)

            layout.addRow(gain_label, gain_slider); layout.addRow(cut_label, cut_slider); layout.addRow(dw_label, dw_slider)
            self.param_widgets += [gain_slider, cut_slider, dw_slider]

        elif isinstance(eff, Chorus):
            depth_label = QtWidgets.QLabel(f"Depth (ms): {eff.depth_ms:.1f}")
            depth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); depth_slider.setRange(1, 50); depth_slider.setValue(int(eff.depth_ms))
            def on_depth(v):
                depth_label.setText(f"Depth (ms): {v}")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.depth_ms = float(v); new_chain[idx] = new_eff; self.swap_chain(new_chain)
            depth_slider.valueChanged.connect(on_depth)

            rate_label = QtWidgets.QLabel(f"Rate (Hz): {eff.rate_hz:.2f}")
            rate_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); rate_slider.setRange(1, 500); rate_slider.setValue(int(eff.rate_hz * 100))
            def on_rate(v):
                rate_label.setText(f"Rate (Hz): {v/100.0:.2f}")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.rate_hz = v / 100.0; new_chain[idx] = new_eff; self.swap_chain(new_chain)
            rate_slider.valueChanged.connect(on_rate)

            mix_label = QtWidgets.QLabel(f"Mix: {eff.mix:.2f}")
            mix_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); mix_slider.setRange(0, 1000); mix_slider.setValue(int(eff.mix * 1000))
            def on_mix(v):
                mix_label.setText(f"Mix: {v/1000.0:.2f}")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.mix = v / 1000.0; new_chain[idx] = new_eff; self.swap_chain(new_chain)
            mix_slider.valueChanged.connect(on_mix)

            layout.addRow(depth_label, depth_slider); layout.addRow(rate_label, rate_slider); layout.addRow(mix_label, mix_slider)
            self.param_widgets += [depth_slider, rate_slider, mix_slider]

        elif isinstance(eff, Delay):
            time_label = QtWidgets.QLabel(f"Time (ms): {eff.time_ms:.0f}")
            time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); time_slider.setRange(1, 2000); time_slider.setValue(int(eff.time_ms))
            def on_time(v):
                time_label.setText(f"Time (ms): {v}")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.time_ms = float(v); new_chain[idx] = new_eff; self.swap_chain(new_chain)
            time_slider.valueChanged.connect(on_time)

            fb_label = QtWidgets.QLabel(f"Feedback: {eff.feedback:.2f}")
            fb_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); fb_slider.setRange(0, 950); fb_slider.setValue(int(eff.feedback * 1000))
            def on_fb(v):
                fb_label.setText(f"Feedback: {v/1000.0:.2f}")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.feedback = v / 1000.0; new_chain[idx] = new_eff; self.swap_chain(new_chain)
            fb_slider.valueChanged.connect(on_fb)

            mix_label = QtWidgets.QLabel(f"Mix: {eff.mix:.2f}")
            mix_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); mix_slider.setRange(0, 1000); mix_slider.setValue(int(eff.mix * 1000))
            def on_mix(v):
                mix_label.setText(f"Mix: {v/1000.0:.2f}")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.mix = v / 1000.0; new_chain[idx] = new_eff; self.swap_chain(new_chain)
            mix_slider.valueChanged.connect(on_mix)

            layout.addRow(time_label, time_slider); layout.addRow(fb_label, fb_slider); layout.addRow(mix_label, mix_slider)
            self.param_widgets += [time_slider, fb_slider, mix_slider]

        elif isinstance(eff, Reverb):
            mix_label = QtWidgets.QLabel(f"Mix: {eff.mix:.2f}")
            mix_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); mix_slider.setRange(0, 1000); mix_slider.setValue(int(eff.mix * 1000))
            def on_mix(v):
                mix_label.setText(f"Mix: {v/1000.0:.2f}")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.mix = v / 1000.0; new_chain[idx] = new_eff; self.swap_chain(new_chain)
            mix_slider.valueChanged.connect(on_mix)

            lp_label = QtWidgets.QLabel(f"LP Cutoff: {eff.lp_cutoff:.0f} Hz")
            lp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); lp_slider.setRange(200, 12000); lp_slider.setValue(int(eff.lp_cutoff))
            def on_lp(v):
                lp_label.setText(f"LP Cutoff: {v} Hz")
                new_chain = list(chain_ref); new_eff = new_chain[idx].clone(); new_eff.lp_cutoff = float(v); new_eff.lp_b0, new_eff.lp_a1 = one_pole_coeffs(new_eff.lp_cutoff, new_eff.sr); new_chain[idx] = new_eff; self.swap_chain(new_chain)
            lp_slider.valueChanged.connect(on_lp)

            layout.addRow(mix_label, mix_slider); layout.addRow(lp_label, lp_slider)
            self.param_widgets += [mix_slider, lp_slider]

    def on_select(self, idx):
        if idx < 0 or idx >= len(chain_ref):
            self.clear_param_area(); return
        eff = chain_ref[idx]; self.build_param_ui_for(eff, idx)

    def swap_chain(self, new_chain):
        global chain_ref
        chain_ref = new_chain

    # Preset UI
    def refresh_preset_list(self):
        self.preset_combo.clear()
        for name in list_presets():
            self.preset_combo.addItem(name)

    def on_save_preset(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        path = save_preset_file(name.strip(), chain_ref)
        self.refresh_preset_list()
        QtWidgets.QMessageBox.information(self, "Saved", f"Preset saved: {os.path.basename(path)}")

    def on_load_preset(self):
        name = self.preset_combo.currentText()
        if not name:
            QtWidgets.QMessageBox.warning(self, "No preset", "No preset selected.")
            return
        path = os.path.join(PRESET_DIR, f"{name}.json")
        try:
            new_chain = load_preset_file(path)
            self.swap_chain(new_chain); self.refresh_chain_list()
            QtWidgets.QMessageBox.information(self, "Loaded", f"Preset loaded: {name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load preset: {e}")

    def on_delete_preset(self):
        name = self.preset_combo.currentText()
        if not name:
            return
        path = os.path.join(PRESET_DIR, f"{name}.json")
        if not os.path.exists(path):
            self.refresh_preset_list(); return
        reply = QtWidgets.QMessageBox.question(self, "Delete Preset", f"Delete preset '{name}'?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                os.remove(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to delete: {e}")
            self.refresh_preset_list()

# ====== 実行 ======
def main():
    global chain_ref
    # 自動ロード: presets フォルダ内で最も新しいプリセットを読み込む
    recent = most_recent_preset_path()
    if recent:
        try:
            new_chain = load_preset_file(recent)
            if new_chain:
                chain_ref = new_chain
                print(f"Auto-loaded preset: {os.path.basename(recent)}")
        except Exception as e:
            print("Failed to auto-load preset:", e)

    print("Available devices:")
    for i, d in enumerate(sd.query_devices()):
        print(i, d['name'], d['max_input_channels'], d['max_output_channels'])
    print("If you want to use a specific device, modify sd.Stream(... device=(in_idx,out_idx) ...).")

    sd.default.device = (14, 13)
    stream = sd.Stream(samplerate=SR, blocksize=BLOCKSIZE, channels=CHANNELS, callback=audio_callback)
    try:
        stream.start()
    except Exception as e:
        print("Failed to start stream:", e); return

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.refresh_preset_list(); win.refresh_chain_list()
    win.show()
    exit_code = app.exec_()

    stream.stop(); stream.close()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()