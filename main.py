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
    """正しいパスワードが入力されたらTrueを返す"""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # ログイン画面の表示
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

# --- 3. データ前処理ロジック ---
def preprocess_data(file, platform):
    try:
        if platform == "MT4 (HTML Report)":
            tables = pd.read_html(file)
            df = tables[0] 
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            returns = pd.to_numeric(df['Profit'], errors='coerce').dropna()
            
        elif platform == "TradingView (CSV)":
            df = pd.read_csv(file)
            target_col = [c for c in df.columns if 'Profit' in c][0]
            returns = pd.to_numeric(df[target_col], errors='coerce').dropna()
            
        elif platform == "MT5 (CSV)":
            df = pd.read_csv(file, sep='\t', encoding='utf-16')
            returns = pd.to_numeric(df['Profit'], errors='coerce').dropna()
            
        else: # カスタム (CSV)
            df = pd.read_csv(file)
            returns = df.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
            
        return returns
    except Exception as e:
        st.error(f"データの読み込みに失敗しました。形式を確認してください: {e}")
        return None

# --- 4. 統計解析エンジン (10,000回モンテカルロ搭載) ---
def analyze_strategy(returns_vec):
    returns = returns_vec.values
    if len(returns) < 10:
        return None
    
    # 基本統計（年率換算シャープレシオ）
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    
    # CPCV的アプローチ (データを5分割して安定性を確認)
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
        
        if i % 250 == 0: # 負荷軽減のため250回ごとに更新
            my_bar.progress(i / mc_iterations, text=f"{progress_text} : {i}/{mc_iterations}")
            
    my_bar.empty()
    p_value = np.sum(np.array(mc_results) >= sharpe) / mc_iterations
    
    return sharpe, pbo_score, p_value, split_sharpes, mc_results

# --- 5. AIコンサルタント機能 (ユーザーのAPIキーを使用) ---
def get_ai_advice(api_key, platform, num_trades, sharpe, pbo, p_val):
    if not api_key:
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        verdict = "合格" if (pbo < 25 and p_val < 0.05) else "注意" if pbo < 50 else "棄却"
        
        prompt = f"""
        あなたはヘッジファンドのシニアアナリストです。以下の統計データに基づき、このトレード戦略の健全性を厳格に診断してください。
        
        【データ】
        プラットフォーム: {platform}
        総トレード数: {num_trades}
        シャープレシオ: {sharpe:.2f}
        PBO (過学習確率): {pbo:.1f}%
        モンテカルロ p値: {p_val:.4f}
        判定: {verdict}
        
        【指示】
        1. なぜこの判定になったのか、PBOとp値の意味を交えて専門的に解説してください。
        2. この戦略が未来の相場で直面する可能性が高いリスクを具体的に指摘してください。
        3. 運用を開始すべきか、ロジックを修正すべきか、具体的なアクションを提案してください。
        
        ※日本語で、プロフェッショナルかつ建設的なトーンで回答してください。
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a professional financial data scientist."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI診断エラー: APIキーが無効か、残高が不足している可能性があります。 ({e})"

# --- 6. メインアプリケーション実行 ---
if check_password():
    st.title("🛡️ プロフェッショナル・戦略健全性診断ツール")
    st.caption("Produced by Singapore Financial IT Lab")

    # サイドバー設定
    with st.sidebar:
        st.header("1. 解析設定")
        platform_choice = st.selectbox(
            "プラットフォームを選択",
            ["TradingView (CSV)", "MT4 (HTML Report)", "MT5 (CSV)", "カスタム (CSV)"]
        )
        uploaded_file = st.file_uploader("バックテストファイルをアップロード", type=['csv', 'html'])
        
        st.divider()
        st.header("2. AI診断オプション")
        user_api_key = st.text_input("OpenAI API Keyを入力", type="password", help="あなたのAPIキーを使用します。サーバーには保存されません。")
        st.caption("※APIキーがなくても統計解析は実行可能です。")

    # メイン表示エリア
    if uploaded_file:
        returns_data = preprocess_data(uploaded_file, platform_choice)
        
        if returns_data is not None and len(returns_data) >= 10:
            # 解析実行
            res = analyze_strategy(returns_data)
            
            if res:
                sharpe, pbo, p_val, split_sharpes, mc_results = res
                
                # --- セクション1: 総合判定 ---
                st.divider()
                if pbo < 25 and p_val < 0.05:
                    st.success("### 判定: 🟢 合格（統計的優位性が認められます）")
                elif pbo < 50:
                    st.warning("### 判定: 🟡 注意（過学習の懸念があります）")
                else:
                    st.error("### 判定: 🔴 棄却（統計的優位性が認められません）")

                # --- セクション2: 統計メトリクス ---
                col1, col2, col3 = st.columns(3)
                col1.metric("期待シャープレシオ", f"{sharpe:.2f}")
                col2.metric("PBO (過学習確率)", f"{pbo:.1f}%")
                col3.metric("モンテカルロ p値", f"{p_val:.4f}")

                # --- セクション3: AIアドバイス ---
                if user_api_key:
                    with st.status("AIコンサルタントが戦略を精査中...", expanded=True):
                        advice = get_ai_advice(user_api_key, platform_choice, len(returns_data), sharpe, pbo, p_val)
                        if advice:
                            st.markdown(advice)
                else:
                    st.info("💡 サイドバーにOpenAI APIキーを入力すると、ここに詳細なAI診断レポートが表示されます。")

                # --- セクション4: 可視化グラフ ---
                st.divider()
                tab1, tab2 = st.tabs(["📊 パフォーマンス分布 (CPCV)", "🎲 偶然性検証 (Monte Carlo)"])
                
                with tab1:
                    fig1 = px.histogram(split_sharpes, nbins=10, 
                                        title="分割データごとのシャープレシオ分布",
                                        labels={'value': 'Sharpe Ratio', 'count': '回数'})
                    st.plotly_chart(fig1, use_container_width=True)
                    st.caption("※分布が右側に寄り、幅が狭いほど、時期を選ばず安定して勝てる強固な手法であることを示します。")

                with tab2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(x=mc_results, name='ランダム(猿)の成績', marker_color='#AAAAAA'))
                    fig2.add_vline(x=sharpe, line_dash="dash", line_color="red", 
                                   annotation_text="あなたの実力", annotation_position="top right")
                    fig2.update_layout(title="モンテカルロ検定：10,000回シミュレーション",
                                      xaxis_title="シャープレシオ", yaxis_title="頻度", barmode='overlay')
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption("※赤線（あなたの手法）が灰色の分布より右側に大きく離れているほど、実力である証拠です。")
        else:
            st.error("有効なトレードデータが不足しています（最低10件以上必要です）。")
    else:
        # 初期画面
        st.info("サイドバーからバックテストファイルをアップロードして解析を開始してください。")
        st.image("https://via.placeholder.com/1000x400.png?text=Waiting+for+Your+Backtest+Data...", use_column_width=True)