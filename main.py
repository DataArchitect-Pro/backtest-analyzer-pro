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
def preprocess_data(file, platform, custom_col=None):
    try:
        # --- HTMLレポート処理 (ストラテジーテスター & リアルトレード両対応) ---
        if platform == "MT4/MT5 (HTML Report)":
            tables = pd.read_html(file)
            best_returns = None
            
            # 全テーブルをスキャンして、最も「トレード履歴らしい」データを持つ表を探す
            for df in tables:
                # 各行をスキャンしてヘッダー（Profit列）を探す
                for i in range(min(20, len(df))): 
                    row_values = [str(val).strip() for val in df.iloc[i]]
                    
                    # 損益を示す可能性のあるキーワードを探す
                    target_keywords = ['Profit', 'Profit/Loss', '利益']
                    found_col_idx = next((idx for idx, val in enumerate(row_values) if val in target_keywords), None)
                    
                    if found_col_idx is not None:
                        temp_df = df.copy()
                        temp_df.columns = row_values
                        temp_df = temp_df.iloc[i+1:] # ヘッダー以降をデータとする
                        
                        # 損益列を数値化
                        col_name = row_values[found_col_idx]
                        rets = pd.to_numeric(temp_df[col_name], errors='coerce').dropna()
                        # 決済データのみ抽出（注文や入出金の0を除外）
                        rets = rets[rets != 0]
                        
                        # 最もデータ件数が多いテーブルを採用（集計表との誤認を避ける）
                        if best_returns is None or len(rets) > len(best_returns):
                            best_returns = rets
            
            if best_returns is None or len(best_returns) == 0:
                st.error("レポート内に有効な損益データが見つかりませんでした。")
                return None
            return best_returns
            
        # --- カスタム CSV 処理 (列選択対応) ---
        elif platform == "カスタム (CSV)":
            df = pd.read_csv(file)
            if custom_col and custom_col in df.columns:
                returns = pd.to_numeric(df[custom_col], errors='coerce').dropna()
            else:
                # 選択されていない場合は数値列の1つ目を採用
                returns = df.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
            return returns
            
        return None
        
    except Exception as e:
        st.error(f"データの解析中にエラーが発生しました: {e}")
        return None

# --- 4. 統計解析エンジン (10,000回モンテカルロ) ---
def analyze_strategy(returns_vec):
    returns = returns_vec.values
    if len(returns) < 10:
        return None
    
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    
    splits = np.array_split(returns, 5)
    split_sharpes = [np.mean(s) / np.std(s) * np.sqrt(252) for s in splits if len(s) > 5 and np.std(s) > 0]
    pbo_score = np.mean([1 if s < sharpe * 0.6 else 0 for s in split_sharpes]) * 100
    
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
        あなたはヘッジファンドのシニアアナリストです。以下の統計データに基づき、この戦略を診断してください。
        プラットフォーム: {platform}, トレード数: {num_trades}, シャープ: {sharpe:.2f}, PBO: {pbo:.1f}%, p値: {p_val:.4f}, 判定: {verdict}
        分析、リスク、アクションプランの3部構成で、プロフェッショナルな日本語で回答してください。
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

    custom_profit_col = None

    with st.sidebar:
        st.header("1. 解析設定")
        platform_choice = st.selectbox(
            "プラットフォームを選択", 
            ["MT4/MT5 (HTML Report)", "カスタム (CSV)"]
        )
        uploaded_file = st.file_uploader("ファイルをアップロード", type=['csv', 'html', 'htm'])
        
        # カスタムCSVの場合の列選択機能
        if uploaded_file and platform_choice == "カスタム (CSV)":
            try:
                # プレビュー読み込み
                df_preview = pd.read_csv(uploaded_file)
                uploaded_file.seek(0) # ポインタを戻す
                custom_profit_col = st.selectbox(
                    "損益(Profit)データが含まれる列を選択してください",
                    df_preview.columns,
                    help="各トレードの損益（金額）が入っている列を選んでください。"
                )
            except Exception as e:
                st.error("CSVの読み込みに失敗しました。")

        st.divider()
        st.header("2. AI診断オプション")
        user_api_key = st.text_input("OpenAI API Keyを入力してください", type="password")
        st.caption("※APIキーなしでも統計解析は可能です。")

    if uploaded_file:
        returns_data = preprocess_data(uploaded_file, platform_choice, custom_profit_col)
        
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
                        status.update(label="✅ AIプロフェッショナル診断が完了しました", state="complete", expanded=True)
                else:
                    st.info("💡 サイドバーにOpenAI APIキーを入力すると、詳細なAI診断レポートが表示されます。")

                st.divider()
                tab1, tab2 = st.tabs(["📊 パフォーマンス分布", "🎲 モンテカルロ検証"])
                with tab1:
                    st.plotly_chart(px.histogram(split_sharpes, nbins=10, title="分割データ別のシャープレシオ分布"), use_container_width=True)
                with tab2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(x=mc_results, name='ランダムな成績分布', marker_color='#AAAAAA'))
                    fig2.add_vline(x=sharpe, line_dash="dash", line_color="red", annotation_text="あなたの実力")
                    fig2.update_layout(title="モンテカルロ検定：10,000回検証結果", barmode='overlay')
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            if returns_data is not None:
                st.error(f"データ件数が不足しています（検出数: {len(returns_data)}件）。解析には最低10件必要です。")
    else:
        st.info("サイドバーからバックテストまたはリアルトレードのレポートをアップロードしてください。")
