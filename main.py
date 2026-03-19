import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import base64

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="Professional Strategy Health Analyzer", layout="wide")

# --- 2. パスワード認証機能 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]: return True
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 購入者専用ログイン")
        password = st.text_input("パスワードを入力してください", type="password")
        if st.button("ログイン"):
            if "ACCESS_PASSWORD" in st.secrets and password == st.secrets["ACCESS_PASSWORD"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("パスワードが正しくありません")
    return False

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
    my_bar = st.progress(0, text="10,000回シミュレーション中...")
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
    prompt = f"""
    あなたは金融機関のクオンツアナリストです。
    トレード数: {len(df)}, シャープ: {sharpe:.2f}, PBO: {pbo:.1f}%, p値: {p_val:.4f}
    判定が合格でない場合、エントリー条件、決済、銘柄選定の観点から『戦略改善ワークシート』として具体的な数学的・論理的アドバイスを提示してください。
    """
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# --- 6. レポート出力機能 ---
def generate_html_report(df, stats, advice):
    sharpe, pbo, p_val = stats
    report_html = f"""
    <html><body>
    <h1>戦略診断レポート</h1>
    <p>期待シャープレシオ: {sharpe:.2f}</p>
    <p>PBO (過学習確率): {pbo:.1f}%</p>
    <p>モンテカルロ p値: {p_val:.4f}</p>
    <hr><h2>AI分析・改善提言</h2>
    <div style="white-space: pre-wrap;">{advice}</div>
    </body></html>
    """
    return report_html

# --- 7. メインUI ---
if check_password():
    st.title("🛡️ Professional Strategy Health Analyzer")
    with st.sidebar:
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
            
            # 判定表示
            st.divider()
            if pbo < 25 and p_val < 0.05: st.success("### 判定: 🟢 合格")
            elif pbo < 50: st.warning("### 判定: 🟡 注意")
            else: st.error("### 判定: 🔴 棄却")

            m1, m2, m3 = st.columns(3)
            m1.metric("期待シャープレシオ", f"{sharpe:.2f}")
            m2.metric("PBO (過学習確率)", f"{pbo:.1f}%")
            m3.metric("モンテカルロ p値", f"{p_val:.4f}")

            # AI診断
            advice = "診断にはAPIキーが必要です。"
            if user_api_key:
                with st.status("AI解析中...", expanded=True) as status:
                    advice = get_ai_advice(user_api_key, data, (sharpe, pbo, p_val))
                    st.markdown(advice)
                    status.update(label="✅ 診断完了", state="complete", expanded=True)
            else:
                st.info("💡 簡易診断: 優位性は統計的に検証されています。" if p_val < 0.05 else "💡 簡易診断: 詳細な改善案はAPIキーを入力してください。")

            # レポート出力
            st.divider()
            html_report = generate_html_report(data, (sharpe, pbo, p_val), advice)
            st.download_button("📜 戦略診断レポート(HTML)をダウンロード", data=html_report, file_name="Strategy_Report.html", mime="text/html")
            
            # 可視化
            t1, t2 = st.tabs(["📊 パフォーマンス分布", "🎲 モンテカルロ検証"])
            with t1: st.plotly_chart(px.histogram(split_sharpes, title="期間別安定性"), use_container_width=True)
            with t2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=mc_results, name='ランダム成績'))
                fig.add_vline(x=sharpe, line_dash="dash", line_color="red", annotation_text="あなたの実力")
                st.plotly_chart(fig, use_container_width=True)
        else: st.error("有効なトレードデータが不足しています。")
