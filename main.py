import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="Pro Backtest Analyzer", layout="wide")

# --- 2. パスワード認証機能 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 購入者専用ログイン")
        password = st.text_input("noteに記載のパスワードを入力してください", type="password")
        if st.button("ログイン"):
            if "ACCESS_PASSWORD" in st.secrets and password == st.secrets["ACCESS_PASSWORD"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("パスワードが正しくありません")
    return False

# --- 3. 強化版 データ前処理ロジック ---
def preprocess_data(file, platform):
    try:
        # --- HTMLレポート処理 (MT4 / MT5 両対応) ---
        if platform == "MT4/MT5 (HTML Report)":
            tables = pd.read_html(file)
            df_trades = None
            
            # 全テーブルをスキャンして損益データが含まれる表を特定
            for table in tables:
                temp_df = table.copy()
                temp_df.columns = temp_df.iloc[0]
                
                # 損益を示す可能性のある列名を検索
                target_cols = ['Profit', 'Profit/Loss', '利益']
                found_col = next((c for c in target_cols if c in temp_df.columns), None)
                
                if found_col:
                    df_trades = temp_df.drop(temp_df.index[0])
                    profit_col_name = found_col
                    break
            
            if df_trades is None:
                st.error("レポート内に損益データ（Profit列）が見つかりませんでした。")
                return None
                
            returns = pd.to_numeric(df_trades[profit_col_name], errors='coerce').dropna()
            # 決済以外の行（注文中など）を除外するため0以外を抽出
            returns = returns[returns != 0]
            
        # --- TradingView CSV 処理 ---
        elif platform == "TradingView (CSV)":
            df = pd.read_csv(file)
            target_col = [c for c in df.columns if 'Profit' in c][0]
            returns = pd.to_numeric(df[target_col], errors='coerce').dropna()
            
        # --- カスタム CSV 処理 ---
        else:
            df = pd.read_csv(file)
            # 数値列の最初のものを取得
            returns = df.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
            
        return returns
        
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        return None

# --- 4. 統計解析エンジン (10,000回モンテカルロ) ---
def analyze_strategy(returns_vec):
    returns = returns_vec.values
    if len(returns) < 10:
        return None
    
    # 期待シャープレシオ (年率換算)
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    
    # CPCV (過学習検証)
    splits = np.array_split(returns, 5)
    split_sharpes = [np.mean(s) / np.std(s) * np.sqrt(252) for s in splits if len(s) > 5 and np.std(s) > 0]
    pbo_score = np.mean([1 if s < sharpe * 0.6 else 0 for s in split_sharpes]) * 100
    
    # モンテカルロ検定 (10,000回シミュレーション)
    mc_iterations = 10000
    mc_results = []
    
    progress_text = "統計的有意性を検証中（10,000回のシミュレーション）..."
    my_bar = st.progress(0, text=progress_text)
    
    for i in range(mc_iterations):
        shuffled = np.random.permutation(returns)
        std_val = np.std(shuffled)
        res = np.mean(shuffled) / std_val * np.sqrt(252) if std_val != 0 else 0
        mc_results.append(res)
        
        if i % 250 == 0:
            my_bar.progress(i / mc_iterations, text=f"{progress_text} : {i}/{mc_iterations} 完了")
            
    my_bar.empty()
    p_value = np.sum(np.array(mc_results) >= sharpe) / mc_iterations
    
    return sharpe, pbo_score, p_value, split_sharpes, mc_results

# --- 5. AIコンサルタント機能 ---
def get_ai_advice(api_key, platform, num_trades, sharpe, pbo, p_val):
    if not api_key: return None
    try:
        client = OpenAI(api_key=api_key)
        verdict = "合格" if (pbo < 25 and p_val < 0.05) else "注意" if pbo < 50 else "棄却"
        prompt = f"""
        あなたはヘッジファンドのシニアアナリストです。
        プラットフォーム: {platform}, トレード数: {num_trades}, シャープ: {sharpe:.2f}, PBO: {pbo:.1f}%, p値: {p_val:.4f}, 判定: {verdict}
        分析、リスク、提言の3部構成で、プロフェッショナルな日本語で回答してください。
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI診断エラー: {e}"

# --- 6. メインアプリケーション ---
if check_password():
    st.title("🛡️ プロフェッショナル・戦略健全性診断ツール")
    st.caption("Produced by Singapore Financial IT Lab")

    with st.sidebar:
        st.header("1. 解析設定")
        platform_choice = st.selectbox(
            "プラットフォームを選択", 
            ["MT4/MT5 (HTML Report)", "TradingView (CSV)", "カスタム (CSV)"]
        )
        uploaded_file = st.file_uploader("ファイルをアップロード", type=['csv', 'html', 'htm'])
        st.divider()
        st.header("2. AI診断オプション")
        user_api_key = st.text_input("OpenAI API Keyを入力してください", type="password")
        st.caption("※APIキーなしでも統計解析は可能です。")

    if uploaded_file:
        returns_data = preprocess_data(uploaded_file, platform_choice)
        
        if returns_data is not None and len(returns_data) >= 10:
            res = analyze_strategy(returns_data)
            
            if res:
                sharpe, pbo, p_val, split_sharpes, mc_results = res
                
                st.divider()
                if pbo < 25 and p_val < 0.05:
                    st.success("### 判定: 🟢 合格（統計的優位性が認められます）")
                elif pbo < 50:
                    st.warning("### 判定: 🟡 注意（過学習の懸念があります）")
                else:
                    st.error("### 判定: 🔴 棄却（統計的優位性が認められません）")

                col1, col2, col3 = st.columns(3)
                col1.metric("期待シャープレシオ", f"{sharpe:.2f}")
                col2.metric("PBO (過学習確率)", f"{pbo:.1f}%")
                col3.metric("モンテカルロ p値", f"{p_val:.4f}")

                # AI診断 (自動展開)
                if user_api_key:
                    with st.status("AIコンサルタントが戦略を精査中...", expanded=True) as status:
                        advice = get_ai_advice(user_api_key, platform_choice, len(returns_data), sharpe, pbo, p_val)
                        if advice:
                            st.markdown(advice)
                        status.update(label="✅ AI診断が完了しました", state="complete", expanded=True)
                else:
                    st.info("💡 サイドバーにOpenAI APIキーを入力すると、詳細なAI診断レポートが表示されます。")

                st.divider()
                tab1, tab2 = st.tabs(["📊 パフォーマンス分布", "🎲 モンテカルロ検証"])
                with tab1:
                    st.plotly_chart(px.histogram(split_sharpes, nbins=10, title="分割データ別のシャープレシオ分布"), use_container_width=True)
                with tab2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(x=mc_results, name='ランダム成績分布', marker_color='#AAAAAA'))
                    fig2.add_vline(x=sharpe, line_dash="dash", line_color="red", annotation_text="あなたの実力")
                    fig2.update_layout(title="モンテカルロ検定：10,000回検証結果", barmode='overlay')
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("有効なトレードデータが不足しています。")
    else:
        st.info("サイドバーからファイルをアップロードしてください。")
