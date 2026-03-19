import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="Professional Strategy Health Analyzer", layout="wide")

# --- 2. パスワード認証機能 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

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

# --- 3. データ前処理ロジック (Open/Close Time, Item対応) ---
def preprocess_data(file, platform, custom_col=None):
    try:
        if platform == "MT4/MT5 (HTML Report)":
            tables = pd.read_html(file)
            df_final = pd.DataFrame()
            
            for df in tables:
                for i in range(min(20, len(df))):
                    row_values = [str(val).strip() for val in df.iloc[i]]
                    
                    # 損益(Profit)列を起点に各列を特定
                    target_keywords = ['Profit', 'Profit/Loss', '利益']
                    p_idx = next((idx for idx, val in enumerate(row_values) if val in target_keywords), None)
                    
                    if p_idx is not None:
                        temp_df = df.copy()
                        temp_df.columns = row_values
                        temp_df = temp_df.iloc[i+1:].reset_index(drop=True)
                        
                        # 必要な列のインデックスを取得
                        col_indices = {
                            'Profit': p_idx,
                            'Open Time': next((j for j, v in enumerate(row_values) if 'Open Time' in v or 'Time' in v), None),
                            'Close Time': next((j for j, v in enumerate(row_values) if 'Close Time' in v), None),
                            'Item': next((j for j, v in enumerate(row_values) if 'Item' in v or 'Symbol' in v or '銘柄' in v), None)
                        }
                        
                        processed = pd.DataFrame()
                        processed['Profit'] = pd.to_numeric(temp_df.iloc[:, col_indices['Profit']], errors='coerce')
                        
                        if col_indices['Open Time'] is not None:
                            processed['Open Time'] = temp_df.iloc[:, col_indices['Open Time']]
                        if col_indices['Close Time'] is not None:
                            processed['Close Time'] = temp_df.iloc[:, col_indices['Close Time']]
                        if col_indices['Item'] is not None:
                            processed['Item'] = temp_df.iloc[:, col_indices['Item']]
                        
                        # 損益が発生している有効なトレードのみ抽出
                        processed = processed[processed['Profit'] != 0].dropna(subset=['Profit'])
                        if len(processed) > len(df_final):
                            df_final = processed
            
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
    
    # CPCV (5分割)
    splits = np.array_split(returns, 5)
    split_sharpes = [np.mean(s) / np.std(s) * np.sqrt(252) for s in splits if len(s) > 5 and np.std(s) > 0]
    pbo_score = np.mean([1 if s < sharpe * 0.6 else 0 for s in split_sharpes]) * 100
    
    # モンテカルロ (10,000回)
    mc_iterations = 10000
    mc_results = []
    my_bar = st.progress(0, text="統計的優位性を検証中（10,000回シミュレーション）")
    for i in range(mc_iterations):
        shuffled = np.random.permutation(returns)
        res = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252) if np.std(shuffled) != 0 else 0
        mc_results.append(res)
        if i % 500 == 0: my_bar.progress(i/mc_iterations)
    my_bar.empty()
    
    p_value = np.sum(np.array(mc_results) >= sharpe) / mc_iterations
    return sharpe, pbo_score, p_value, split_sharpes, mc_results

# --- 5. AI診断 & 戦略改善ワークシート生成 ---
def get_ai_advice(api_key, df, stats):
    client = OpenAI(api_key=api_key)
    sharpe, pbo, p_val = stats
    item_stats = df['Item'].value_counts().to_dict() if 'Item' in df.columns else "不明"
    
    prompt = f"""
    あなたは金融機関のシニア・クオンツアナリストです。
    トレード数: {len(df)}, シャープ: {sharpe:.2f}, PBO: {pbo:.1f}%, p値: {p_val:.4f}
    銘柄分布: {item_stats}
    
    【指示】
    1. 統計データの詳細分析（なぜこの結果になったか）。
    2. もし判定が「注意」や「棄却」の場合、改善のための『戦略改善ワークシート』を具体的なステップで作成してください。
    エントリー条件、損切り幅、銘柄選定、時間軸の観点から数学的根拠に基づき提言すること。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# --- 6. メインUI ---
if check_password():
    st.title("🛡️ プロフェッショナル・戦略健全性診断ツール")
    
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
        st.header("2. AI診断設定 (任意)")
        user_api_key = st.text_input("OpenAI API Keyを入力", type="password")
        st.caption("※APIキー入力で「戦略改善ワークシート」が自動生成されます。")

    if uploaded_file:
        data = preprocess_data(uploaded_file, platform_choice, custom_profit_col)
        
        if data is not None and len(data) >= 10:
            # 統計解析実行
            sharpe, pbo, p_val, split_sharpes, mc_results = analyze_strategy(data)
            
            # --- 判定表示 (APIキーの有無に関わらず表示) ---
            st.divider()
            if pbo < 25 and p_val < 0.05:
                st.success("### 判定: 🟢 合格（統計的優位性が極めて高いです）")
                status_color = "success"
            elif pbo < 50:
                st.warning("### 判定: 🟡 注意（過学習、または優位性の欠如が疑われます）")
                status_color = "warning"
            else:
                st.error("### 判定: 🔴 棄却（統計的優位性が認められません。ロジックの抜本的修正が必要です）")
                status_color = "error"

            m1, m2, m3 = st.columns(3)
            m1.metric("期待シャープレシオ", f"{sharpe:.2f}")
            m2.metric("PBO (過学習確率)", f"{pbo:.1f}%")
            m3.metric("モンテカルロ p値 (10,000回)", f"{p_val:.4f}")

            with st.expander("詳細なトレードデータ（Open/Close/Item）を表示"):
                st.dataframe(data, use_container_width=True)

            # --- AI診断 / 簡易診断 ---
            st.subheader("🧐 戦略診断・改善レポート")
            if user_api_key:
                with st.status("AIコンサルタントが戦略改善ワークシートを生成中...", expanded=True) as status:
                    advice = get_ai_advice(user_api_key, data, (sharpe, pbo, p_val))
                    st.markdown(advice)
                    status.update(label="✅ 戦略診断レポートの生成が完了しました", state="complete", expanded=True)
            else:
                # APIキーがない場合の簡易診断
                if status_color == "success":
                    st.info("💡 【簡易診断】現在のロジックは非常に強固です。銘柄分散を検討しつつ、運用継続を推奨します。")
                elif status_color == "warning":
                    st.info("💡 【簡易診断】特定の期間や銘柄に依存している可能性があります。バックテスト期間を延ばして再検証してください。")
                else:
                    st.info("💡 【簡易診断】この戦略は『偶然』による利益の可能性が高いです。エントリー条件の厳格化が必要です。詳細な改善案が必要な場合はAPIキーを入力してください。")

            # --- 可視化 ---
            st.divider()
            t1, t2 = st.tabs(["📊 パフォーマンス安定性分布", "🎲 偶然性検証 (10,000回)"])
            with t1:
                st.plotly_chart(px.histogram(split_sharpes, nbins=10, title="分割データ別のパフォーマンス分布"), use_container_width=True)
            with t2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=mc_results, name='ランダムな成績', marker_color='#AAAAAA'))
                fig.add_vline(x=sharpe, line_dash="dash", line_color="red", annotation_text="あなたの実力")
                fig.update_layout(title="モンテカルロ検定結果", barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("有効なデータが不足しています（最低10件以上必要）。")
    else:
        st.info("サイドバーからレポート（HTMLまたはCSV）をアップロードしてください。")
