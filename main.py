import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from datetime import datetime

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

# --- 3. 超強化版 データ前処理ロジック (時刻・銘柄抽出対応) ---
def preprocess_data(file, platform, custom_col=None):
    try:
        if platform == "MT4/MT5 (HTML Report)":
            tables = pd.read_html(file)
            df_final = pd.DataFrame()
            
            # 全テーブルをスキャン
            for df in tables:
                for i in range(min(15, len(df))):
                    row_values = [str(val).strip() for val in df.iloc[i]]
                    
                    # 損益列(Profit)を探す
                    target_keywords = ['Profit', 'Profit/Loss', '利益']
                    p_idx = next((idx for idx, val in enumerate(row_values) if val in target_keywords), None)
                    
                    if p_idx is not None:
                        temp_df = df.copy()
                        temp_df.columns = row_values
                        temp_df = temp_df.iloc[i+1:].reset_index(drop=True)
                        
                        # 列の特定 (MT4/MT5/Statement/Testerの差異を吸収)
                        col_map = {
                            'Profit': p_idx,
                            'Open Time': next((j for j, v in enumerate(row_values) if 'Open Time' in v or 'Time' in v), None),
                            'Close Time': next((j for j, v in enumerate(row_values) if 'Close Time' in v), None),
                            'Item': next((j for j, v in enumerate(row_values) if 'Item' in v or 'Symbol' in v), None)
                        }
                        
                        processed = pd.DataFrame()
                        processed['Profit'] = pd.to_numeric(temp_df.iloc[:, col_map['Profit']], errors='coerce')
                        
                        if col_map['Open Time'] is not None:
                            processed['Open Time'] = temp_df.iloc[:, col_map['Open Time']]
                        if col_map['Close Time'] is not None:
                            processed['Close Time'] = temp_df.iloc[:, col_map['Close Time']]
                        if col_map['Item'] is not None:
                            processed['Item'] = temp_df.iloc[:, col_map['Item']]
                        
                        # 損益が発生している有効なトレード行のみ抽出
                        processed = processed[processed['Profit'] != 0].dropna(subset=['Profit'])
                        if len(processed) > len(df_final):
                            df_final = processed
            
            if df_final.empty:
                st.error("有効なトレードデータが見つかりませんでした。")
                return None
            return df_final
            
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
    if len(returns) < 10: return None
    
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
        if i % 500 == 0: my_bar.progress(i/mc_iterations)
    my_bar.empty()
    
    p_value = np.sum(np.array(mc_results) >= sharpe) / mc_iterations
    return sharpe, pbo_score, p_value, split_sharpes, mc_results

# --- 5. AIアドバイス & 戦略改善ワークフロー ---
def get_ai_diagnosis(api_key, df, stats, detailed=False):
    if not api_key: return None
    client = OpenAI(api_key=api_key)
    sharpe, pbo, p_val = stats
    
    # 銘柄・時間の傾向を要約
    item_summary = df['Item'].value_counts().to_dict() if 'Item' in df.columns else "不明"
    
    prompt = f"""
    あなたは世界最高峰の金融データサイエンティストです。
    【統計データ】
    トレード数: {len(df)}, シャープ: {sharpe:.2f}, PBO: {pbo:.1f}%, p値: {p_val:.4f}
    銘柄分布: {item_summary}
    
    【指示】
    {"詳細診断と戦略改善プランを提示してください。" if detailed else "簡易的な診断結果を提示してください。"}
    特にもし判定が「棄却(p値>0.05)」または「注意(PBO>30%)」の場合、以下の観点で具体的かつ数学的なロジック改善策を「戦略改善ワークシート」として出力してください：
    1. エントリー条件の厳格化
    2. 保有時間と利益の相関から見た決済ロジックの修正
    3. 銘柄選定の見直し
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
        uploaded_file = st.file_uploader("レポートをアップロード", type=['csv', 'html', 'htm'])
        
        custom_col = None
        if uploaded_file and platform_choice == "カスタム (CSV)":
            df_p = pd.read_csv(uploaded_file)
            custom_col = st.selectbox("損益列を選択", df_p.columns)
            uploaded_file.seek(0)

        st.divider()
        st.header("2. AI診断設定")
        user_key = st.text_input("OpenAI API Key (任意)", type="password")
        st.info("APIキー入力で「詳細診断＆戦略改善プラン」が解放されます。")

    if uploaded_file:
        data = preprocess_data(uploaded_file, platform_choice, custom_col)
        
        if data is not None and len(data) >= 10:
            # 統計解析
            sharpe, pbo, p_val, split_sharpes, mc_results = analyze_strategy(data)
            
            # --- セクション1: 統計ダッシュボード ---
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("期待シャープレシオ", f"{sharpe:.2f}")
            m2.metric("PBO (過学習確率)", f"{pbo:.1f}%")
            m3.metric("モンテカルロ p値", f"{p_val:.4f}")

            # --- セクション2: データプレビュー (Open/Close/Item) ---
            with st.expander("抽出されたトレードデータ詳細"):
                st.dataframe(data, use_container_width=True)

            # --- セクション3: AI診断 & 戦略改善 ---
            st.subheader("🤖 AIプロフェッショナル診断")
            if user_key:
                with st.status("詳細解析 & 改善プランを生成中...", expanded=True) as status:
                    advice = get_ai_diagnosis(user_key, data, (sharpe, pbo, p_val), detailed=True)
                    st.markdown(advice)
                    status.update(label="✅ 戦略改善ワークシートが生成されました", state="complete", expanded=True)
            else:
                # 簡易診断（ロジックによる自動生成）
                if p_val < 0.05 and pbo < 30:
                    st.success("【簡易診断】この戦略は統計的に極めて堅牢です。実運用への移行を推奨します。")
                else:
                    st.warning("【簡易診断】優位性が不足、または過学習の疑いがあります。詳細な改善案が必要な場合はAPIキーを入力してください。")

            # --- セクション4: 可視化 ---
            st.divider()
            t1, t2 = st.tabs(["📊 パフォーマンス安定性", "🎲 偶然性検証"])
            with t1:
                st.plotly_chart(px.histogram(split_sharpes, nbins=10, title="期間別安定性分布"), use_container_width=True)
            with t2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=mc_results, name='ランダム成績', marker_color='#AAAAAA'))
                fig.add_vline(x=sharpe, line_dash="dash", line_color="red", annotation_text="あなたの実力")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("データが不足しています。")
