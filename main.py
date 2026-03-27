import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import base64
import time
import uuid

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="プロフェッショナル戦略健全性診断ツール", layout="wide")

# ==========================================
# 2. 認証ロジック (先勝ちブロック + タイムアウト機能)
# ==========================================
# 💡 アプリ全体で共有される「ログイン中のセッション状態」
@st.cache_resource
def get_active_sessions():
    # 構造: { "user_id": {"token": "...", "last_active": 1690000000.0} }
    return {}

def check_password():
    common_password = st.secrets.get("APP_PASSWORD", "vUtZ7&Lyk!XuMS4r)G")
    allowed_ids = st.secrets.get("ALLOWED_IDS", ["a380.rolls.royce@gmail.com"])
    
    active_sessions = get_active_sessions()
    current_time = time.time()
    
    # 💡 安全装置：30分（1800秒）操作がなかったセッションは「ログアウト忘れ（ブラウザ閉じ）」とみなし、ロックを解除する
    TIMEOUT_SECONDS = 1800 
    for uid in list(active_sessions.keys()):
        if current_time - active_sessions[uid]["last_active"] > TIMEOUT_SECONDS:
            del active_sessions[uid]

    # セッションステートの初期化
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "session_token" not in st.session_state:
        st.session_state["session_token"] = None

    current_user = st.session_state["user_id"]
    current_token = st.session_state["session_token"]

    # 💡 現在ログイン中のユーザーが操作した際の生存確認（タイムアウトの更新）
    if current_user:
        if current_user in active_sessions and active_sessions[current_user]["token"] == current_token:
            # 操作するたびに寿命をリセット（延長）する
            active_sessions[current_user]["last_active"] = current_time
        else:
            # タイムアウト等でサーバーから消去された場合はログアウト状態に戻す
            st.session_state["user_id"] = None
            st.session_state["session_token"] = None
            current_user = None

    # ログインしていない場合の画面表示
    if not current_user:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #333; font-size: 2.5em;'>🔒 ユーザーログイン</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; font-size: 1.3em; margin-bottom: 10px;'>付与された専用IDと、共通パスワードを入力してください。</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #d32f2f; font-size: 0.9em; margin-bottom: 30px;'>※同時ログイン不可（別の人が使用中のIDではログインできません）</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                st.markdown("<div style='font-size: 1.0em; font-weight: bold; margin-bottom: 4px; color: #333;'>📝 専用ユーザーID</div>", unsafe_allow_html=True)
                user_id = st.text_input("ID", placeholder="例：noteの注文IDなど", label_visibility="collapsed")
                
                st.markdown("<div style='font-size: 1.0em; font-weight: bold; margin-top: 12px; margin-bottom: 4px; color: #333;'>🔑 共通パスワード</div>", unsafe_allow_html=True)
                password = st.text_input("パスワード", type="password", placeholder="記事内にあるパスワードを入力", label_visibility="collapsed")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("ログイン", use_container_width=True)

                if submit:
                    if password != common_password:
                        st.error("❌ パスワードが間違っています。")
                    elif user_id not in allowed_ids:
                        st.error("❌ 登録されていないユーザーIDです。")
                    elif user_id in active_sessions:
                        # 💡 ここで「後からのログイン」を完全にブロックする
                        st.error("❌ このIDは現在別の端末・ブラウザで利用中です。（同時ログイン不可）\n\n※前の利用者がログアウトするか、一定時間（最大30分）経過するまでお待ちください。")
                    else:
                        # 認証成功：新しいユニークトークンを発行し、時間を記録
                        new_token = str(uuid.uuid4())
                        st.session_state["user_id"] = user_id
                        st.session_state["session_token"] = new_token
                        active_sessions[user_id] = {"token": new_token, "last_active": current_time}
                        st.rerun() 
        
        # 認証されるまで以降のコードを一切実行しない
        st.stop()

# アプリ起動時に必ずパスワードとセッションをチェック
check_password()

# --- 3. データ前処理 (Open/Close Time, Item対応) ---
def preprocess_data(file, platform, custom_col=None):
    try:
        if platform == "MT4/MT5 (HTML Report)":
            tables = pd.read_html(file)
            df_final = pd.DataFrame()
            for df in tables:
                for i in range(min(20, len(df))):
                    row_values = [str(val).strip() for val in df.iloc[i]]
                    target_keywords = ['Profit', 'Profit/Loss', '利益']
                    p_idx = next((idx for idx, val in enumerate(row_values) if val in target_keywords), None)
                    if p_idx is not None:
                        temp_df = df.copy()
                        temp_df.columns = row_values
                        temp_df = temp_df.iloc[i+1:].reset_index(drop=True)
                        col_indices = {
                            'Profit': p_idx,
                            'Open Time': next((j for j, v in enumerate(row_values) if 'Open Time' in v or 'Time' in v), None),
                            'Close Time': next((j for j, v in enumerate(row_values) if 'Close Time' in v), None),
                            'Item': next((j for j, v in enumerate(row_values) if 'Item' in v or 'Symbol' in v or '銘柄' in v), None)
                        }
                        processed = pd.DataFrame()
                        processed['Profit'] = pd.to_numeric(temp_df.iloc[:, col_indices['Profit']], errors='coerce')
                        if col_indices['Open Time'] is not None: processed['Open Time'] = temp_df.iloc[:, col_indices['Open Time']]
                        if col_indices['Close Time'] is not None: processed['Close Time'] = temp_df.iloc[:, col_indices['Close Time']]
                        if col_indices['Item'] is not None: processed['Item'] = temp_df.iloc[:, col_indices['Item']]
                        processed = processed[processed['Profit'] != 0].dropna(subset=['Profit'])
                        if len(processed) > len(df_final): df_final = processed
            return df_final if not df_final.empty else None
        elif platform == "カスタム (CSV)":
            df = pd.read_csv(file)
            if custom_col:
                df['Profit'] = pd.to_numeric(df[custom_col], errors='coerce')
                return df.dropna(subset=['Profit'])
        return None
    except Exception as e:
        st.error(f"解析エラー: {e}")
        return None

# --- 4. 統計解析エンジン ---
def analyze_strategy(df):
    returns = df['Profit'].values
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    splits = np.array_split(returns, 5)
    split_sharpes = [np.mean(s) / np.std(s) * np.sqrt(252) for s in splits if len(s) > 5 and np.std(s) > 0]
    pbo_score = np.mean([1 if s < sharpe * 0.6 else 0 for s in split_sharpes]) * 100
    mc_iterations = 10000
    mc_results = []
    my_bar = st.progress(0, text="10,000回モンテカルロ検定中...")
    for i in range(mc_iterations):
        shuffled = np.random.permutation(returns)
        res = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252) if np.std(shuffled) != 0 else 0
        mc_results.append(res)
        if i % 1000 == 0: my_bar.progress(i/mc_iterations)
    my_bar.empty()
    p_value = np.sum(np.array(mc_results) >= sharpe) / mc_iterations
    return sharpe, pbo_score, p_value, split_sharpes, mc_results

# --- 5. AI診断機能 ---
def get_ai_advice(api_key, df, stats):
    client = OpenAI(api_key=api_key)
    sharpe, pbo, p_val = stats
    item_stats = df['Item'].value_counts().to_dict() if 'Item' in df.columns else "データなし"
    prompt = f"""
    あなたは金融機関のクオンツアナリストです。
    トレード数: {len(df)}, シャープ: {sharpe:.2f}, PBO: {pbo:.1f}%, p値: {p_val:.4f}
    銘柄分布: {item_stats}
    上記に基づき、戦略の優位性を診断し、具体的な改善案を『戦略改善ワークシート』として提示してください。
    """
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# --- 6. レポート出力機能 (グラフ埋め込み版) ---
def generate_html_report(df, stats, advice, fig1, fig2):
    sharpe, pbo, p_val = stats
    fig1_html = fig1.to_html(full_html=False, include_plotlyjs='cdn')
    fig2_html = fig2.to_html(full_html=False, include_plotlyjs='cdn')
    
    report_html = f"""
    <html>
    <head><meta charset="utf-8"><title>戦略診断レポート</title></head>
    <body style="font-family: sans-serif; padding: 20px;">
        <h1>🛡️ 戦略診断・改善レポート</h1>
        <hr>
        <h3>■ 統計スコア</h3>
        <ul>
            <li>期待シャープレシオ: {sharpe:.2f}</li>
            <li>PBO (過学習確率): {pbo:.1f}%</li>
            <li>モンテカルロ p値: {p_val:.4f}</li>
        </ul>
        <hr>
        <h3>■ AI診断・改善ワークシート</h3>
        <div style="background: #f4f4f4; padding: 15px; white-space: pre-wrap;">{advice}</div>
        <hr>
        <h3>■ 解析グラフ</h3>
        <div>{fig1_html}</div>
        <div style="margin-top:20px;">{fig2_html}</div>
    </body>
    </html>
    """
    return report_html

# --- 7. メインUI ---
st.title("🛡️ プロフェッショナル戦略健全性診断ツール")
st.caption("Produced by Singapore Financial IT Lab")

with st.sidebar:
    # 💡 追加: ログアウト機能
    st.markdown(f"**👤 ログイン中: {st.session_state['user_id']}**")
    if st.button("ログアウト", use_container_width=True):
        active_sessions = get_active_sessions()
        if st.session_state["user_id"] in active_sessions:
            del active_sessions[st.session_state["user_id"]]
        st.session_state["user_id"] = None
        st.session_state["session_token"] = None
        st.rerun()
    st.divider()

    st.header("1. 解析設定")
    platform_choice = st.selectbox("プラットフォーム", ["MT4/MT5 (HTML Report)", "カスタム (CSV)"])
    uploaded_file = st.file_uploader("レポートファイルをアップロード", type=['csv', 'html', 'htm'])
    custom_profit_col = None
    if uploaded_file and platform_choice == "カスタム (CSV)":
        df_p = pd.read_csv(uploaded_file)
        custom_profit_col = st.selectbox("損益(Profit)列を選択", df_p.columns)
        uploaded_file.seek(0)
    st.divider()
    st.header("2. AI診断設定")
    user_api_key = st.text_input("OpenAI API Key (任意)", type="password")

if uploaded_file:
    data = preprocess_data(uploaded_file, platform_choice, custom_profit_col)
    if data is not None and len(data) >= 10:
        sharpe, pbo, p_val, split_sharpes, mc_results = analyze_strategy(data)
        
        st.divider()
        if pbo < 25 and p_val < 0.05: st.success("### 判定: 🟢 合格")
        elif pbo < 50: st.warning("### 判定: 🟡 注意")
        else: st.error("### 判定: 🔴 棄却")

        m1, m2, m3 = st.columns(3)
        m1.metric("期待シャープレシオ", f"{sharpe:.2f}")
        m2.metric("PBO (過学習確率)", f"{pbo:.1f}%")
        m3.metric("モンテカルロ p値", f"{p_val:.4f}")

        # AI診断
        advice = "診断レポートの生成にはAPIキーが必要です。"
        if user_api_key:
            with st.status("AI解析 & ワークシート生成中...", expanded=True) as status:
                advice = get_ai_advice(user_api_key, data, (sharpe, pbo, p_val))
                st.markdown(advice)
                status.update(label="✅ 診断レポート生成完了", state="complete", expanded=True)
        else:
            st.info("💡 簡易判定: 統計的優位性は検証済みです。詳細アドバイスはAPIキーを入力してください。")

        # グラフ生成
        fig1 = px.histogram(split_sharpes, nbins=10, title="パフォーマンス安定性（CPCV分布）")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=mc_results, name='ランダム', marker_color='#AAAAAA'))
        fig2.add_vline(x=sharpe, line_dash="dash", line_color="red", annotation_text="あなたの実力")
        fig2.update_layout(title="モンテカルロ検定（1万回）")

        # レポート出力
        st.divider()
        html_report = generate_html_report(data, (sharpe, pbo, p_val), advice, fig1, fig2)
        st.download_button("📜 戦略診断レポート(HTML)を保存", data=html_report, file_name="Strategy_Report.html", mime="text/html")
        
        t1, t2 = st.tabs(["📊 パフォーマンス分布", "🎲 モンテカルロ検証"])
        with t1: st.plotly_chart(fig1, use_container_width=True)
        with t2: st.plotly_chart(fig2, use_container_width=True)
    else: st.error("有効なトレードデータが不足しています（最低10件）。")
