"""
음성 분석 및 포먼트 편집기
의존성: pip install streamlit parselmouth numpy scipy soundfile matplotlib plotly streamlit-plotly-events
실행: streamlit run Formant.py
"""

import io
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal
import soundfile as sf
import streamlit as st

# ── 한글 폰트 설정 (Windows / Linux / macOS 모두 대응) ──────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path


def _set_korean_font():
    # 1) 레포에 번들된 폰트를 최우선으로 사용 (가장 안정적)
    candidates = [
        Path(__file__).parent / "fonts" / "NanumGothic.ttf",          # 번들 폰트
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),       # packages.txt: fonts-nanum
    ]
    for path in candidates:
        if path.exists():
            fm.fontManager.addfont(str(path))
            matplotlib.rcParams["font.family"] = fm.FontProperties(fname=str(path)).get_name()
            break
    else:
        # 2) 시스템에 이미 설치된 한글 폰트 탐색 (로컬 개발 환경 대비)
        installed = {f.name for f in fm.fontManager.ttflist}
        for name in ["NanumGothic", "Malgun Gothic", "AppleGothic", "NanumBarunGothic"]:
            if name in installed:
                matplotlib.rcParams["font.family"] = name
                break

    matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


_set_korean_font()
st.set_page_config(page_title="음성 분석기", layout="wide")
st.title("음성 분석 및 포먼트 편집기")
st.info(
    "**단모음**을 **한 번만** 발음한 음성을 사용하세요. "
)

# ── 오디오 입력 ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("오디오 입력")
    input_mode = st.radio("입력 방식", ["마이크 녹음", "파일 업로드"])
    st.divider()
    st.header("분석 설정")
    gender = st.radio(
        "성별",
        ["남성 (ceiling 5000 Hz)", "여성 (ceiling 5500 Hz)"],
        help="성별에 따라 포먼트의 가능한 최대 주파수를 설정합니다.",
    )
    formant_ceiling = 5000 if gender.startswith("남성") else 5500
    sg_window_ms = st.slider("스펙트로그램 윈도우 (ms)", 5, 50, 25, 1,
                              help="한번의 푸리에 분석 시간입니다. 길수록 주파수 해상도가, 짧을수록 시간 해상도가 높습니다.")
    sg_window_s  = sg_window_ms / 1000.0

sound: parselmouth.Sound | None = None
tmp_path: str | None = None

if input_mode == "파일 업로드":
    with st.sidebar:
        uploaded = st.file_uploader("WAV 파일 업로드", type=["wav"])
    if uploaded:
        raw = uploaded.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(raw)
            tmp_path = f.name
        sound = parselmouth.Sound(tmp_path)
else:
    with st.sidebar:
        rec = st.audio_input("마이크 녹음 (버튼을 눌러 시작/정지)", key="audio_rec")
        if rec is not None:
            if st.button("재녹음", help="현재 녹음을 지우고 다시 녹음합니다"):
                del st.session_state["audio_rec"]
                st.rerun()
    if rec:
        raw = rec.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(raw)
            tmp_path = f.name
        sound = parselmouth.Sound(tmp_path)

if sound is None:
    st.info("왼쪽 사이드바에서 WAV 파일을 업로드하거나 마이크로 녹음하세요.")
    st.stop()

sr = int(sound.sampling_frequency)
nyq = sr // 2

st.subheader("원본 음성")
st.audio(tmp_path)

# ── LPC 유틸리티 ──────────────────────────────────────────────────────────────
def _levinson(r: np.ndarray, order: int) -> np.ndarray:
    """Levinson-Durbin recursion → LPC 계수 (길이 order+1, a[0]=1.0)"""
    a = np.zeros(order)
    e = r[0]
    for i in range(order):
        if e < 1e-14:
            break
        lam = -(np.dot(a[:i], r[i:0:-1]) + r[i + 1])
        k = lam / e
        k = np.clip(k, -1 + 1e-9, 1 - 1e-9)
        a_new = a.copy()
        a_new[i] = k
        if i > 0:
            a_new[:i] += k * a[i - 1::-1]
        a = a_new
        e *= 1 - k ** 2
    return np.r_[1.0, a]


@st.cache_data
def compute_lpc(audio_bytes: bytes, sr: int) -> np.ndarray:
    """음성 바이트열 → LPC 계수 (캐시됨)"""
    x = np.frombuffer(audio_bytes, dtype=np.float64).copy()
    x -= x.mean()

    # 중앙 구간 최대 4096 샘플 사용 (단모음의 안정적인 부분)
    center = len(x) // 2
    half = min(len(x) // 2, 2048)
    segment = x[max(0, center - half): center + half]

    win = np.hanning(len(segment))
    x_w = segment * win

    order = min(2 + int(sr / 1000), len(x_w) // 4)

    # FFT 기반 자기상관
    fft_n = 2 ** int(np.ceil(np.log2(2 * len(x_w))))
    X = np.fft.rfft(x_w, n=fft_n)
    r = np.fft.irfft(np.abs(X) ** 2).real[: order + 1]

    return _levinson(r, order)


def lpc_to_formants(lpc_coeffs: np.ndarray, sr: int,
                    n: int = 4, ceiling: int = 5500):
    """LPC 계수 → 포먼트 주파수(Hz), 대역폭(Hz), 복소 루트"""
    roots = np.roots(lpc_coeffs)
    roots = roots[np.imag(roots) > 0]           # 상반부(양의 주파수)만
    freqs = np.angle(roots) * sr / (2 * np.pi)
    bws   = -np.log(np.abs(roots)) * sr / np.pi
    mask  = (freqs > 90) & (freqs < ceiling) & (bws > 0) & (bws < 600)
    freqs, bws, roots = freqs[mask], bws[mask], roots[mask]
    idx = np.argsort(freqs)
    freqs, bws, roots = freqs[idx], bws[idx], roots[idx]
    return freqs[:n], bws[:n], roots[:n]


def shift_formants(lpc_coeffs: np.ndarray,
                   orig_roots: np.ndarray,
                   new_freqs,
                   sr: int) -> np.ndarray:
    """각 포먼트 루트의 각도만 교체하여 새 LPC 계수 반환"""
    all_roots = np.roots(lpc_coeffs).copy()

    for orig_root, new_f in zip(orig_roots, new_freqs):
        new_angle = 2 * np.pi * float(new_f) / sr
        radius    = np.abs(orig_root)
        new_root  = radius * np.exp(1j * new_angle)

        i_pos = int(np.argmin(np.abs(all_roots - orig_root)))
        all_roots[i_pos] = new_root
        i_neg = int(np.argmin(np.abs(all_roots - np.conj(orig_root))))
        all_roots[i_neg] = np.conj(new_root)

    new_lpc = np.poly(all_roots).real
    new_lpc /= new_lpc[0]
    return new_lpc


def resynthesize(audio: np.ndarray,
                 orig_lpc: np.ndarray,
                 new_lpc: np.ndarray) -> np.ndarray:
    """원본 LPC로 잔차 추출 → 새 LPC로 재합성"""
    residual = signal.lfilter(orig_lpc, [1.0], audio)
    synth    = signal.lfilter([1.0], new_lpc, residual)
    peak = np.max(np.abs(synth))
    if peak > 0:
        synth = synth / peak * 0.9
    return synth.astype(np.float32)


# ── 스펙트로그램 공용 데이터 ──────────────────────────────────────────────────
spec  = sound.to_spectrogram(window_length=sg_window_s, maximum_frequency=min(nyq, 8000))
X_sg  = spec.x_grid()
Y_sg  = spec.y_grid()
sg_db = 10 * np.log10(spec.values + 1e-10)

# ─────────────────────────────────────────────────────────────────────────────
tab_sg, tab_fmt, tab_filt, tab_pwr = st.tabs([
    "스펙트로그램",
    "포먼트 편집",
    "주파수 대역 필터",
    "파워 스펙트럼",
])

# ─── Tab 1: 스펙트로그램 ──────────────────────────────────────────────────────
with tab_sg:
    col_ctrl, col_plot = st.columns([1, 3])

    with col_ctrl:
        max_f    = st.number_input("최대 표시 주파수 (Hz)",
                             min_value=1000, max_value=8000, value=5000, step=2)
        show_fmt = st.checkbox("포먼트 표시 (F1–F3)", value=True)
        n_fmt    = st.slider("추출 포먼트 수", 3, 5, 5) if show_fmt else 5

    with col_plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.pcolormesh(X_sg, Y_sg, sg_db, shading="auto", cmap="Greys",
                           vmin=sg_db.max() - 70, vmax=sg_db.max())
        fig.colorbar(im, ax=ax, label="강도 (dB)")

        if show_fmt:
            try:
                fmt_obj = call(sound, "To Formant (burg)", 0, n_fmt, formant_ceiling, 0.025, 50)
                ts = np.linspace(sound.xmin + 0.02, sound.xmax - 0.02, 250)
                for fi, color in zip(range(1, 4), ["deepskyblue", "lime", "magenta"]):
                    vals = [call(fmt_obj, "Get value at time", fi, t, "Hertz", "Linear")
                            for t in ts]
                    vals = [np.nan if (v != v) else v for v in vals]
                    ax.plot(ts, vals, color=color, lw=1.5,
                            label=f"F{fi}", alpha=0.85)
                ax.legend(loc="upper right", fontsize=8)
            except Exception as e:
                st.warning(f"포먼트 추출 실패: {e}")

        ax.set_xlim(sound.xmin, sound.xmax)
        ax.set_ylim(0, max_f)
        ax.set_xlabel("시간 (s)")
        ax.set_ylabel("주파수 (Hz)")
        ax.set_title("스펙트로그램")
        st.pyplot(fig)
        plt.close(fig)

# ─── Tab 4: 파워 스펙트럼 ─────────────────────────────────────────────────────
with tab_pwr:
    t_sel  = st.slider("분석 시점 (s)",
                       float(sound.xmin), float(sound.xmax),
                       float((sound.xmin + sound.xmax) / 2), 0.001)
    win_ms = st.slider("윈도우 (ms)", 10, 100, 25, 1)
    win_s  = win_ms / 1000.0

    fig_ref, ax_ref = plt.subplots(figsize=(10, 2))
    ax_ref.pcolormesh(X_sg, Y_sg, sg_db, shading="auto", cmap="Greys",
                      vmin=sg_db.max() - 70, vmax=sg_db.max())
    ax_ref.axvline(t_sel, color="yellow", lw=1.5, label=f"t={t_sel:.3f}s")
    ax_ref.set_xlim(sound.xmin, sound.xmax)
    ax_ref.set_xlabel("시간 (s)")
    ax_ref.set_ylabel("주파수 (Hz)")
    ax_ref.legend(loc="upper right", fontsize=8)
    st.pyplot(fig_ref)
    plt.close(fig_ref)

    start = max(sound.xmin, t_sel - win_s / 2)
    end   = min(sound.xmax, start + win_s)

    if end - start < 0.005:
        st.warning("분석 구간이 너무 짧습니다.")
    else:
        chunk  = sound.extract_part(start, end, parselmouth.WindowShape.HANNING, 1, False)
        sp     = chunk.to_spectrum()
        freqs2 = sp.xs()
        pwr_db = 10 * np.log10(sp.values[0] ** 2 + sp.values[1] ** 2 + 1e-10)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(freqs2, pwr_db, color="steelblue", lw=1)
        ax2.set_xlim(0, min(8000, nyq))
        ax2.set_xlabel("주파수 (Hz)")
        ax2.set_ylabel("파워 (dB)")
        ax2.set_title(f"파워 스펙트럼   t = {t_sel:.3f} s   (윈도우 {win_ms} ms)")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

# ─── Tab 2: 포먼트 편집 ──────────────────────────────────────────────────────
with tab_fmt:
    st.markdown(
        "슬라이더를 움직이면 해당 포먼트만 이동된 음성과 스펙트로그램이 업데이트됩니다."
    )

    audio_np   = sound.values[0].astype(np.float64)
    lpc_coeffs = compute_lpc(audio_np.tobytes(), sr)
    det_freqs, det_bws, det_roots = lpc_to_formants(lpc_coeffs, sr, n=4, ceiling=formant_ceiling)

    if len(det_freqs) < 2:
        st.warning("포먼트를 충분히 추출하지 못했습니다. "
                   "단모음을 1초 이상 발음하여 다시 시도하세요.")
        st.stop()

    n_det  = len(det_freqs)
    COLORS = ["cyan", "lime", "tomato", "orange"]

    FORMANT_RANGES = [
        (200,  1100),
        (600,  3000),
        (1500, 3500),
        (2500, 4500),
    ]

    detected_vals = [int(round(f)) for f in det_freqs]

    # 슬라이더 session_state 사전 초기화 (미설정 키만)
    # → st.rerun() 없이 개별 초기화할 수 있게, 다른 슬라이더 state 보존
    for i, dv in enumerate(detected_vals):
        if f"fmt_slider_{i}" not in st.session_state:
            st.session_state[f"fmt_slider_{i}"] = dv

    # ── 전체 초기화 버튼 ─────────────────────────────────────────────────────
    _, col_rstall = st.columns([5, 1])
    with col_rstall:
        if st.button("전체 초기화", key="fmt_reset_all"):
            for i, dv in enumerate(detected_vals):
                st.session_state[f"fmt_slider_{i}"] = dv

    # ── 포먼트별 슬라이더 ────────────────────────────────────────────────────
    new_freqs = []
    slider_cols = st.columns(n_det)
    for i, (freq, bw, col) in enumerate(zip(det_freqs, det_bws, slider_cols)):
        if i < len(FORMANT_RANGES):
            lo, hi = FORMANT_RANGES[i]
        else:
            lo, hi = int(freq * 0.5), int(freq * 1.8)
        lo = min(lo, int(freq) - 50)
        hi = max(hi, int(freq) + 50)
        lo = max(lo, 80)
        hi = min(hi, nyq - 100)
        with col:
            lbl_col, rst_col = st.columns([3, 1])
            with lbl_col:
                st.markdown(f"**F{i+1}** <span style='color:{COLORS[i]}'>●</span>",
                            unsafe_allow_html=True)
            with rst_col:
                # st.rerun() 제거: 버튼 클릭 자체가 rerun을 유발하며,
                # 루프 내 st.rerun() 호출 시 이후 슬라이더 state가 소거되는 버그 방지
                if st.button("↺", key=f"fmt_reset_{i}", help="탐지값으로 초기화"):
                    st.session_state[f"fmt_slider_{i}"] = detected_vals[i]
            val = st.slider(
                f"F{i+1}", lo, hi, detected_vals[i], 1,
                key=f"fmt_slider_{i}",
                label_visibility="collapsed",
            )
            st.caption(f"탐지: **{freq:.0f} Hz** | BW: {bw:.0f} Hz")
            new_freqs.append(val)

    # ── 재합성 ──────────────────────────────────────────────────────────────
    all_reset = (new_freqs == detected_vals)
    new_lpc   = shift_formants(lpc_coeffs, det_roots, new_freqs, sr)
    synth     = resynthesize(audio_np, lpc_coeffs, new_lpc)

    st.subheader("수정된 음성")
    if all_reset:
        st.audio(tmp_path)
    else:
        buf = io.BytesIO()
        sf.write(buf, synth, sr, format="WAV", subtype="FLOAT")
        buf.seek(0)
        st.audio(buf, format="audio/wav")

    # ── 스펙트로그램 비교 (항상 표시) ────────────────────────────────────────
    synth_sound = parselmouth.Sound(synth.astype(np.float64), sr)
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax3, snd3, fmts, ttl in zip(
        axes3,
        [sound,     synth_sound],
        [det_freqs, new_freqs],
        ["원본",    "포먼트 수정"],
    ):
        sg3    = snd3.to_spectrogram(window_length=sg_window_s, maximum_frequency=5000)
        X3, Y3 = sg3.x_grid(), sg3.y_grid()
        db3    = 10 * np.log10(sg3.values + 1e-10)
        ax3.pcolormesh(X3, Y3, db3, shading="auto", cmap="Greys",
                       vmin=db3.max() - 70, vmax=db3.max())
        for fi, (fval, color) in enumerate(zip(fmts, COLORS)):
            ax3.axhline(fval, color=color, lw=1.5, linestyle="--",
                        label=f"F{fi+1} = {fval:.0f} Hz")
        ax3.legend(loc="upper right", fontsize=7)
        ax3.set_title(ttl)
        ax3.set_xlabel("시간 (s)")
    axes3[0].set_ylabel("주파수 (Hz)")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# ─── Tab 3: 주파수 대역 필터 ──────────────────────────────────────────────────
with tab_filt:
    st.markdown("선택한 주파수 대역만 남기고 나머지를 제거한 뒤 재생합니다.")
    st.caption("*300-3000Hz 대역은 지상선(landline) 전화의 대략적인 대역폭입니다.")

    col_flow, col_fhigh = st.columns(2)
    with col_flow:
        f_low  = st.slider("하한 주파수 (Hz)", 0, nyq - 100, 300, 10)
    with col_fhigh:
        f_high = st.slider("상한 주파수 (Hz)", 100, nyq, min(3000, nyq-100), 10)

    if f_low >= f_high:
        st.error("하한 주파수는 상한 주파수보다 낮아야 합니다.")
        st.stop()

    # 슬라이더 변경마다 즉시 계산 → 재생바 상시 표시
    raw_np = sound.values[0].copy()
    nyq_f  = sr / 2.0

    if f_low <= 20:
        b, a = signal.butter(6, f_high / nyq_f, btype="low")
    elif f_high >= nyq - 20:
        b, a = signal.butter(6, f_low / nyq_f, btype="high")
    else:
        b, a = signal.butter(6, [f_low / nyq_f, f_high / nyq_f], btype="band")

    filtered = signal.filtfilt(b, a, raw_np)
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered = filtered / peak * 0.9

    buf2 = io.BytesIO()
    sf.write(buf2, filtered.astype(np.float32), sr, format="WAV", subtype="FLOAT")
    buf2.seek(0)

    st.subheader(f"선택 대역 음성  ({f_low}–{f_high} Hz)")
    st.audio(buf2, format="audio/wav")

    # 스펙트로그램 + 선택 대역 표시
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.pcolormesh(X_sg, Y_sg, sg_db, shading="auto", cmap="Greys",
                   vmin=sg_db.max() - 70, vmax=sg_db.max())
    ax4.axhspan(f_low, f_high, alpha=0.25, color="cyan")
    ax4.axhline(f_low,  color="cyan", lw=1.5, linestyle="--", label=f"하한 {f_low} Hz")
    ax4.axhline(f_high, color="lime", lw=1.5, linestyle="--", label=f"상한 {f_high} Hz")
    ax4.set_xlim(sound.xmin, sound.xmax)
    ax4.set_xlabel("시간 (s)")
    ax4.set_ylabel("주파수 (Hz)")
    ax4.set_title(f"선택 대역: {f_low} – {f_high} Hz")
    ax4.legend(loc="upper right", fontsize=8)
    st.pyplot(fig4)
    plt.close(fig4)


    # with st.expander("평균 파워 스펙트럼 보기"):
    #     sp_full     = sound.to_spectrum()
    #     freqs_full  = sp_full.xs()
    #     pwr_full_db = 10 * np.log10(
    #         sp_full.values[0] ** 2 + sp_full.values[1] ** 2 + 1e-10)

    #     fig5, ax5 = plt.subplots(figsize=(10, 3))
    #     ax5.plot(freqs_full, pwr_full_db, color="steelblue", lw=0.8)
    #     ax5.axvspan(f_low, f_high, alpha=0.2, color="cyan", label="선택 대역")
    #     ax5.set_xlim(0, min(8000, nyq))
    #     ax5.set_xlabel("주파수 (Hz)")
    #     ax5.set_ylabel("파워 (dB)")
    #     ax5.set_title("파워 스펙트럼 (전체 구간)")
    #     ax5.legend(fontsize=8)
    #     ax5.grid(alpha=0.3)
    #     st.pyplot(fig5)
    #     plt.close(fig5)

    with st.expander("필터링 후 스펙트로그램 보기"):
        filt_sound = parselmouth.Sound(filtered.astype(np.float64), sr)
        sg_f = filt_sound.to_spectrogram(window_length=sg_window_s,
                                          maximum_frequency=min(nyq, 8000))
        Xf, Yf = sg_f.x_grid(), sg_f.y_grid()
        dbf    = 10 * np.log10(sg_f.values + 1e-10)

        fig6, ax6 = plt.subplots(figsize=(10, 4))
        ax6.pcolormesh(Xf, Yf, dbf, shading="auto", cmap="Greys",
                       vmin=dbf.max() - 70, vmax=dbf.max())
        ax6.axhspan(f_low, f_high, alpha=0.15, color="cyan")
        ax6.set_xlabel("시간 (s)")
        ax6.set_ylabel("주파수 (Hz)")
        ax6.set_title(f"필터링 후 스펙트로그램 ({f_low}–{f_high} Hz)")
        st.pyplot(fig6)
        plt.close(fig6)
