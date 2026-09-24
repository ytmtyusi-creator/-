import os
import re
import numpy as np
import librosa
import streamlit as st
import yt_dlp

# ページ基本設定
st.set_page_config(page_title="現代バズり度 診断ツール", page_icon="🎵")

# タイトル表示
st.title("🎵 現代バズり度 診断ツール v2.0")
st.write(
    "YouTubeのURLを入力すると、2020〜2025年のヒット曲データ（黄金範囲）と照らし合わせてバズり度を測定します！"
)

# 黄金範囲データ
GOLDEN_RANGES = {
    "再生時間": (209, 243.5, 291),
    "BPM": (96, 120, 141),
    "音の明るさ": (2555.5, 2877, 3124.25),
    "音の迫力": (0.206375, 0.248, 0.2869),
    "アタック感": (0.051375, 0.0592, 0.068),
}


# 適合率計算ロジック
def calc_feature_score_v2(key, x, q1, m, q3):
    if key == "BPM":
        pct_raw = calc_raw_pct(x, q1, m, q3, is_unlimited_upper=False)
        pct_double = calc_raw_pct(x * 2, q1, m, q3, is_unlimited_upper=True)
        pct = max(pct_raw, pct_double)
        return round(pct, 1), round((pct / 100) * 20, 2)

    is_unlimited = key in ["音の迫力", "アタック感", "音の明るさ"]
    pct = calc_raw_pct(x, q1, m, q3, is_unlimited_upper=is_unlimited)
    return round(pct, 1), round((pct / 100) * 20, 2)


def calc_raw_pct(x, q1, m, q3, is_unlimited_upper=False):
    if x < m:
        rate = max(0.0, (x - q1) / (m - q1))
    else:
        if is_unlimited_upper:
            rate = 1.0
        else:
            rate = max(0.0, (q3 - x) / (q3 - m))
    return rate * 100


# 403ブロック回避に特化した音声取得関数
def extract_audio_features(youtube_url):
    output_filename = "temp_audio.wav"

    # YouTube側のブロックを完全回避するクライアント偽装設定
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "temp_audio.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        # iOSアプリ・Androidアプリの通信に偽装して403を回避
        "extractor_args": {
            "youtube": {
                "player_client": ["ios", "mweb"],
                "skip": ["dash", "hls"],
            }
        },
    }

    try:
        # ダウンロード実行
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # librosaで読み込み＆解析
        y, sr = librosa.load(output_filename, sr=None)

        duration = round(librosa.get_duration(y=y, sr=sr))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(tempo))
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness = round(float(np.mean(centroid)))
        rms = librosa.feature.rms(y=y)
        power = round(float(np.mean(rms)), 4)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        attack = round(float(np.mean(zcr)), 4)

        return {
            "再生時間": duration,
            "BPM": bpm,
            "音の明るさ": brightness,
            "音の迫力": power,
            "アタック感": attack,
        }

    except Exception as e:
        st.error(f"詳細エラー情報: {e}")
        return None

    finally:
        for file in os.listdir():
            if file.startswith("temp_audio"):
                try:
                    os.remove(file)
                except:
                    pass


# UI部分
url_input = st.text_input("YouTube動画のURLを入力してください")

if st.button("バズり度を判定する！"):
    if not url_input:
        st.warning("URLを入力してください。")
    else:
        with st.spinner("楽曲を解析中...（10〜30秒ほどかかります）"):
            features = extract_audio_features(url_input)

        if features:
            st.success("解析が完了しました！")
            st.subheader("📊 診断レポート")

            total_score = 0
            for key, (q1, m, q3) in GOLDEN_RANGES.items():
                val = features[key]
                pct, score = calc_feature_score_v2(key, val, q1, m, q3)
                total_score += score

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**【{key}】** 実測値: `{val}` (中央値: {m})")
                with col2:
                    st.write(f"適合率: **{pct}%** ({score}/20点)")

            total_score = round(total_score, 1)
            st.divider()

            st.metric(label="🎯 最終バズり度スコア", value=f"{total_score} / 100点")

            if total_score >= 80:
                st.balloons()
                st.success(
                    "評価: 【 Sランク 】現代のヒット曲ストライクゾーンに完璧に合致しています！"
                )
            elif total_score >= 60:
                st.info("評価: 【 Aランク 】SNS発のバズが十分に狙える楽曲です。")
            elif total_score >= 40:
                st.warning(
                    "評価: 【 Bランク 】一部の要素がトレンドから外れています。"
                )
            else:
                st.error("評価: 【 Cランク 】ターゲットを絞ったアプローチ向きです。")
