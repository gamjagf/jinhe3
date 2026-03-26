# ============================================================
# 📊 제품 판매 데이터 분석 & 예측 대시보드
# ============================================================
# streamlit run app.py 로 실행
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 페이지 설정 (가장 먼저!)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="📊 판매 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# 한글 폰트 설정
# ─────────────────────────────────────────
def set_korean_font():
    try:
        font_list = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = ['NanumGothic','Malgun Gothic','AppleGothic','NanumBarunGothic','Noto Sans KR']
        for font in korean_fonts:
            if font in font_list:
                plt.rcParams['font.family'] = font
                break
        else:
            plt.rcParams['font.family'] = 'DejaVu Sans'
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# ─────────────────────────────────────────
# CSS 스타일
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* 전체 배경 */
    .main { background-color: #f8fafc; }

    /* KPI 카드 */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 10px;
    }
    .kpi-value { font-size: 28px; font-weight: 800; color: #1a202c; margin: 8px 0; }
    .kpi-label { font-size: 13px; color: #718096; font-weight: 500; }

    /* 섹션 제목 */
    .section-title {
        font-size: 18px; font-weight: 700;
        color: #2d3748; margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* 안내 박스 */
    .info-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border: 1px solid #667eea40;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
    }

    /* 예측 결과 박스 */
    .predict-box {
        background: linear-gradient(135deg, #11998e15, #38ef7d15);
        border: 1px solid #11998e40;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 샘플 데이터 생성
# ─────────────────────────────────────────
def make_sample_data():
    """샘플 CSV 데이터를 만들어서 반환합니다."""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=12, freq='MS').strftime('%Y-%m').tolist()
    products = ['사과', '바나나', '오렌지', '포도']
    rows = []
    base = {'사과':120, '바나나':85, '오렌지':60, '포도':95}
    for p in products:
        for d in dates:
            rows.append({'날짜': d, '제품명': p, '판매량': int(base[p] + np.random.randint(-20,30))})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────
# 데이터 로드 함수
# ─────────────────────────────────────────
def load_data(uploaded_file):
    """업로드된 CSV 파일을 읽어서 검증합니다."""
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')

        # 필수 컬럼 확인
        required = ['날짜', '제품명', '판매량']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"❌ 필수 컬럼이 없습니다: {missing}\n\n컬럼명을 확인해주세요: 날짜, 제품명, 판매량")
            return None

        df['판매량'] = pd.to_numeric(df['판매량'], errors='coerce').fillna(0).astype(int)
        df['날짜'] = df['날짜'].astype(str)
        return df

    except UnicodeDecodeError:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp949')
            df['판매량'] = pd.to_numeric(df['판매량'], errors='coerce').fillna(0).astype(int)
            return df
        except:
            st.error("❌ 파일을 읽을 수 없습니다. CSV 파일인지 확인해주세요.")
            return None
    except Exception as e:
        st.error(f"❌ 오류가 발생했습니다: {str(e)}")
        return None

# ─────────────────────────────────────────
# 예측 함수 (선형 추세 기반)
# ─────────────────────────────────────────
def predict_sales(sales_list, months):
    """과거 판매량을 보고 앞으로 N개월을 예측합니다."""
    x = np.arange(len(sales_list))
    # 선형 회귀 (추세선)
    z = np.polyfit(x, sales_list, 1)
    p = np.poly1d(z)
    # 미래 X값
    future_x = np.arange(len(sales_list), len(sales_list) + months)
    predicted = [max(0, int(p(xi))) for xi in future_x]
    return predicted

# ─────────────────────────────────────────
# 그래프 그리기 함수들
# ─────────────────────────────────────────
def draw_bar_chart(df):
    """제품별 총 판매량 막대 그래프"""
    product_total = df.groupby('제품명')['판매량'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#667eea','#764ba2','#f093fb','#f5576c','#4facfe','#43e97b']
    bars = ax.bar(product_total.index, product_total.values,
                  color=colors[:len(product_total)], width=0.6, edgecolor='none')
    # 막대 위에 숫자 표시
    for bar, val in zip(bars, product_total.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('제품별 총 판매량', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('판매량 (개)', fontsize=11)
    ax.set_xlabel('제품명', fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig

def draw_trend_chart(product_df, product_name):
    """선택한 제품의 월별 판매 추이 선 그래프"""
    monthly = product_df.groupby('날짜')['판매량'].sum().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(monthly)), monthly.values,
            marker='o', color='#667eea', linewidth=2.5, markersize=7,
            markerfacecolor='white', markeredgewidth=2)
    ax.fill_between(range(len(monthly)), monthly.values, alpha=0.12, color='#667eea')
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=45, ha='right', fontsize=9)
    ax.set_title(f'[{product_name}] 월별 판매 추이', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('판매량 (개)', fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig

def draw_predict_chart(monthly, predicted, product_name):
    """실제 + 예측 데이터를 합쳐서 보여주는 그래프"""
    n_actual = len(monthly)
    n_pred   = len(predicted)
    all_vals = list(monthly.values) + predicted

    fig, ax = plt.subplots(figsize=(12, 5))
    # 실제 데이터
    ax.plot(range(n_actual), monthly.values,
            marker='o', color='#667eea', linewidth=2.5, markersize=7,
            label='실제 판매량', markerfacecolor='white', markeredgewidth=2)
    # 예측 데이터 (점선)
    pred_x = range(n_actual-1, n_actual+n_pred)
    pred_y = [monthly.values[-1]] + predicted
    ax.plot(pred_x, pred_y,
            marker='s', color='#f5576c', linewidth=2.5, markersize=7,
            linestyle='--', label='예측 판매량', markerfacecolor='white', markeredgewidth=2)
    # 구분선
    ax.axvline(x=n_actual-1, color='#cbd5e0', linestyle=':', linewidth=1.5)
    ax.fill_betweenx([0, max(all_vals)*1.15], n_actual-1, n_actual+n_pred-1,
                     alpha=0.05, color='#f5576c')
    ax.set_title(f'[{product_name}] 실제 판매량 + 미래 예측', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('판매량 (개)', fontsize=11)
    ax.legend(fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════
# 메인 앱 시작
# ═══════════════════════════════════════════
def main():

    # ── 상단 헤더 ──
    st.markdown("## 📊 제품 판매 데이터 분석 & 예측 대시보드")
    st.markdown("CSV 파일을 업로드하면 판매 현황을 분석하고 미래 판매량을 예측합니다.")
    st.divider()

    # ─────────────────────────────────────────
    # 사이드바
    # ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        st.markdown("---")

        # 파일 업로드
        st.markdown("**📂 CSV 파일 업로드**")
        uploaded_file = st.file_uploader(
            "파일을 선택하세요",
            type=['csv'],
            help="날짜, 제품명, 판매량 컬럼이 필요합니다"
        )

        # 샘플 데이터 버튼
        use_sample = st.button("🎲 샘플 데이터로 시작", use_container_width=True)

        st.markdown("---")
        st.markdown("**📋 CSV 파일 형식 안내**")
        st.code("날짜,제품명,판매량\n2025-01,사과,120\n2025-02,사과,135\n2025-01,바나나,80", language='text')
        st.caption("※ 컬럼명을 정확히 맞춰주세요")

    # ─────────────────────────────────────────
    # 데이터 준비
    # ─────────────────────────────────────────
    df = None

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success(f"✅ 파일 업로드 완료! 총 {len(df):,}행의 데이터가 로드되었습니다.")

    elif use_sample or 'df_sample' in st.session_state:
        df = make_sample_data()
        st.session_state['df_sample'] = True
        st.info("🎲 샘플 데이터를 사용 중입니다. 왼쪽에서 실제 CSV를 업로드하세요.")

    # ─────────────────────────────────────────
    # 데이터가 없을 때 안내 화면
    # ─────────────────────────────────────────
    if df is None:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            <div class="info-box" style="text-align:center; padding:40px;">
                <div style="font-size:60px; margin-bottom:16px;">📂</div>
                <h3 style="color:#2d3748;">시작하는 방법</h3>
                <br/>
                <p>👈 왼쪽 사이드바에서<br/>
                <strong>CSV 파일을 업로드</strong>하거나<br/>
                <strong>샘플 데이터 버튼</strong>을 눌러보세요!</p>
                <br/>
                <p style="color:#718096; font-size:13px;">
                필요한 컬럼: 날짜 / 제품명 / 판매량
                </p>
            </div>
            """, unsafe_allow_html=True)
        return

    # ─────────────────────────────────────────
    # 제품 선택 (사이드바)
    # ─────────────────────────────────────────
    products = sorted(df['제품명'].unique().tolist())
    with st.sidebar:
        st.markdown("---")
        st.markdown("**🏷️ 분석할 제품 선택**")
        selected = st.selectbox("제품을 선택하세요", products)
        st.markdown("---")
        st.markdown("**📅 예측 기간 설정**")
        future_months = st.slider("예측할 개월 수", min_value=1, max_value=12, value=3)

    # ─────────────────────────────────────────
    # KPI 카드 (핵심 지표)
    # ─────────────────────────────────────────
    st.markdown('<div class="section-title">📈 핵심 지표</div>', unsafe_allow_html=True)

    total   = df['판매량'].sum()
    avg     = int(df['판매량'].mean())
    best    = df.groupby('제품명')['판매량'].sum().idxmax()
    best_val= df.groupby('제품명')['판매량'].sum().max()
    count   = len(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 총 판매량", f"{total:,}개")
    with col2:
        st.metric("📊 평균 판매량", f"{avg:,}개")
    with col3:
        st.metric("🏆 최고 판매 제품", f"{best}")
    with col4:
        st.metric("📋 데이터 건수", f"{count:,}건")

    st.divider()

    # ─────────────────────────────────────────
    # 데이터 미리보기
    # ─────────────────────────────────────────
    with st.expander("🔍 데이터 미리보기 (클릭해서 펼치기)", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"전체 {len(df):,}행 중 최대 20행만 표시")

    # ─────────────────────────────────────────
    # 제품별 총 판매량 그래프
    # ─────────────────────────────────────────
    st.markdown('<div class="section-title">📊 제품별 판매 분석</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        fig_bar = draw_bar_chart(df)
        st.pyplot(fig_bar)
        plt.close()
        st.caption("💡 막대가 높을수록 총 판매량이 많은 제품입니다.")

    with col_right:
        # 제품별 판매량 표
        product_summary = df.groupby('제품명')['판매량'].agg(['sum','mean','max','min']).reset_index()
        product_summary.columns = ['제품명','총판매량','평균','최대','최소']
        product_summary = product_summary.sort_values('총판매량', ascending=False)
        product_summary['총판매량'] = product_summary['총판매량'].apply(lambda x: f"{x:,}")
        product_summary['평균']    = product_summary['평균'].apply(lambda x: f"{int(x):,}")
        st.dataframe(product_summary, use_container_width=True, hide_index=True)
        st.caption("제품별 판매량 요약 통계")

    st.divider()

    # ─────────────────────────────────────────
    # 선택한 제품 월별 추이
    # ─────────────────────────────────────────
    st.markdown(f'<div class="section-title">📈 [{selected}] 월별 판매 추이</div>', unsafe_allow_html=True)

    product_df = df[df['제품명'] == selected]
    monthly    = product_df.groupby('날짜')['판매량'].sum().sort_index()

    fig_trend = draw_trend_chart(product_df, selected)
    st.pyplot(fig_trend)
    plt.close()

    # 추이 해석
    if len(monthly) >= 2:
        first_half = monthly.values[:len(monthly)//2].mean()
        second_half= monthly.values[len(monthly)//2:].mean()
        diff = second_half - first_half
        if diff > 5:
            trend_msg = f"📈 **{selected}**의 판매량은 전반적으로 **증가 추세**입니다. 지속적인 인기를 확인할 수 있어요!"
        elif diff < -5:
            trend_msg = f"📉 **{selected}**의 판매량은 전반적으로 **감소 추세**입니다. 마케팅 강화가 필요할 수 있어요."
        else:
            trend_msg = f"➡️ **{selected}**의 판매량은 전반적으로 **안정적인 수준**을 유지하고 있습니다."
        st.info(trend_msg)

    st.divider()

    # ─────────────────────────────────────────
    # 미래 판매량 예측
    # ─────────────────────────────────────────
    st.markdown(f'<div class="section-title">🔮 [{selected}] 향후 {future_months}개월 판매 예측</div>', unsafe_allow_html=True)

    predicted = predict_sales(monthly.values.tolist(), future_months)

    # 예측 그래프
    fig_pred = draw_predict_chart(monthly, predicted, selected)
    st.pyplot(fig_pred)
    plt.close()
    st.caption("📌 실선(파란색) = 실제 판매량 / 점선(빨간색) = 예측 판매량")

    # 예측 결과 표
    st.markdown("**예측 결과 상세**")
    last_date = monthly.index[-1]
    try:
        pred_dates = pd.date_range(
            pd.to_datetime(last_date) + pd.DateOffset(months=1),
            periods=future_months, freq='MS'
        ).strftime('%Y-%m').tolist()
    except:
        pred_dates = [f"예측{i+1}개월후" for i in range(future_months)]

    pred_df = pd.DataFrame({'예측 월': pred_dates, '예측 판매량': [f"{v:,}개" for v in predicted]})
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

    # 예측 요약
    avg_pred = int(np.mean(predicted))
    avg_real = int(monthly.mean())
    change   = avg_pred - avg_real
    sign     = "+" if change >= 0 else ""
    st.markdown(f"""
    <div class="predict-box">
        🔮 <strong>예측 요약</strong><br/><br/>
        현재 평균 판매량: <strong>{avg_real:,}개</strong><br/>
        예측 평균 판매량: <strong>{avg_pred:,}개</strong><br/>
        변화: <strong>{sign}{change:,}개 ({sign}{int(change/avg_real*100) if avg_real>0 else 0}%)</strong>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("📊 이 앱은 선형 추세 기반 예측을 사용합니다. 실제 비즈니스 의사결정에는 추가 분석이 필요할 수 있습니다.")

# 앱 실행
if __name__ == '__main__':
    main()
