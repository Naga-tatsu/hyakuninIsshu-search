import pandas as pd
import gradio as gr
import re
import os
import random

# ==========================
# 共通処理
# ==========================
def load_csv(file_path="OguraHyakuninIshu.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    return pd.read_csv(file_path)

def normalize_kana(text):
    if not isinstance(text, str):
        return ""
    return text.replace("ゐ","い").replace("ゑ","え").replace("　","").replace(" ","")

def normalize_number_text(text):
    text = text.translate(str.maketrans("０１２３４５６７８９","0123456789"))
    for a,b in [("−","-"),("〜","~"),("～","~"),("、",","),("　"," ")]:
        text = text.replace(a,b)
    return text

def parse_numbers(input_text):
    input_text = normalize_number_text(input_text)
    numbers = set()
    for part in re.split(r"[,\s]+", input_text):
        if not part: continue
        if "-" in part or "~" in part:
            try:
                start,end = map(int,re.split(r"[-~]", part))
                # start <= end を保証
                if start > end: start, end = end, start
                numbers.update(range(start,end+1))
            except: continue
        else:
            try: numbers.add(int(part))
            except: continue
    return sorted(numbers)

def split_keywords(raw_text):
    if not raw_text: return []
    # タグはリストで渡されるため、raw_textがリストの場合は結合してから処理
    if isinstance(raw_text, list):
        raw_text = ",".join(raw_text)
    return [normalize_kana(k) for k in re.split(r"[,\s、]+", raw_text) if k]

def format_search_summary(df_res, text_kw="", author_kw="", number_kw="", tags=None):
    """
    検索条件と結果件数のサマリーHTMLを生成。
    df_resは条件AND/ORでフィルタリングされた最終結果。
    """
    tags = tags or []
    summaries = []
    
    # 歌番号のサマリーは、検索条件に入力があった場合のみ表示
    numbers = parse_numbers(number_kw) if number_kw else []
    
    if text_kw:
        summaries.append(f'本文「{text_kw}」')
    if author_kw:
        summaries.append(f'作者「{author_kw}」')
    if number_kw and numbers:
        # 入力された歌番号リストに基づいて表示
        summaries.append(f'歌番号{",".join(map(str,numbers))}')
    if tags:
        # tagsがリストの場合、その要素を表示
        summaries.append(f'ジャンル「{",".join(tags)}」')
        
    condition_str = "、".join(summaries)
    
    if condition_str:
        return f"【条件：{condition_str}】 該当する歌：{len(df_res)}首"
    return f"該当する歌：{len(df_res)}首"

# ==========================
# 検索関数 (変更なし)
# ==========================
def search_poems(df, text_kw_raw="", author_kw_raw="", number_kw_raw="", tags_raw=None):
    tags_raw = tags_raw or []
    text_kws = split_keywords(text_kw_raw)
    author_kws = split_keywords(author_kw_raw)
    # tags_rawはリストとして渡されるので、split_keywordsの内部で結合して処理
    special_kws = split_keywords(tags_raw) 
    number_list = parse_numbers(number_kw_raw) if number_kw_raw else []

    df_norm = df.copy()
    # 比較対象の列をひらがなノーマライズ
    for col in ['歌人（ひらがな）','上の句（ひらがな）','下の句（ひらがな）']:
        if col in df_norm.columns:
            df_norm[col] = df_norm[col].astype(str).apply(normalize_kana)

    # OR検索用のマスク
    masks = []

    # 本文検索
    if text_kws:
        text_mask = pd.Series(False, index=df.index)
        for kw in text_kws:
            text_mask |= (
                df_norm.get('上の句','').astype(str).str.contains(kw) |
                df_norm.get('下の句','').astype(str).str.contains(kw) |
                df_norm.get('上の句（ひらがな）','').astype(str).str.contains(kw) |
                df_norm.get('下の句（ひらがな）','').astype(str).str.contains(kw)
            )
        masks.append(text_mask)

    # 作者検索
    if author_kws:
        author_mask = pd.Series(False, index=df.index)
        special_map = {
            "上皇":["上皇"],
            "僧":["僧","僧侶","坊主","坊さん","お坊さん","坊"]
        }
        for kw in author_kws:
            # 特殊キーワードの対応
            if kw in special_map["上皇"]:
                # 「上皇」列が1のものを検索 (列が存在する場合)
                author_mask |= df.get("上皇",pd.Series(0, index=df.index))==1
            elif kw in special_map["僧"]:
                # 「僧」列が1のものを検索 (列が存在する場合)
                author_mask |= df.get("僧",pd.Series(0, index=df.index))==1
            else:
                # 歌人名（ひらがな/漢字）での部分一致検索
                author_mask |= df_norm.get('歌人（ひらがな）','').astype(str).str.contains(kw) | df.get('歌人','').astype(str).str.contains(kw)
        masks.append(author_mask)

    # 歌番号検索
    if number_list and 'No' in df.columns:
        # No列を数値に変換（エラー時はNaN）
        df_norm['No_int'] = pd.to_numeric(df['No'], errors='coerce')
        number_mask = df_norm['No_int'].isin(number_list)
        masks.append(number_mask)

    # タグ検索
    if special_kws:
        tag_mask = pd.Series(False, index=df.index)
        for kw in special_kws:
            # タグ列が存在し、値が1のものを検索
            if kw in df.columns:
                tag_mask |= df[kw]==1
        masks.append(tag_mask)

    # OR検索：どれか1つでも True なら含む
    if masks:
        # 全てのマスクを結合（OR条件）
        mask = pd.Series(False, index=df.index)
        for m in masks:
            mask |= m
    else:
        # 条件なしなら全件
        mask = pd.Series(True, index=df.index)

    return df[mask].reset_index(drop=True)


# ==========================
# 変体仮名 (変更なし)
# ==========================
hentaigana_map = {
    "あ":["阿"],"う":["有"],"お":["於"],"か":["可"],"き":["幾","支"],"く":["具"],"け":["介"],"こ":["古"],
    "さ":["佐"],"す":["春","寿"],"せ":["勢"],"そ":["所","楚"],"た":["多","堂"],"ち":["遅"],"つ":["徒"],"と":["登"],
    "な":["那"],"に":["尓","耳"],"ね":["年"],"の":["能"],"は":["者","盤"],"ひ":["非","悲"],"ふ":["婦","布"],"ほ":["本"],
    "ま":["万","満"],"み":["見","身"],"む":["無"],"め":["免"],"も":["裳"],"や":["夜"],"ゆ":["由"],"よ":["世"], 
    "ら":["羅"],"り":["里"],"る":["流","累"],"れ":["連"],"わ":["王"]
}

def random_hentaigana(text, mapping, prob=0.3):
    """テキストをランダムな確率で変体仮名（対応する漢字の最初の1文字）に変換"""
    return "".join([random.choice(mapping.get(c, [c]))[0] if c in mapping and random.random()<prob else c for c in text])

# ==========================
# 共通レンダラー（学習用ベース）
# ==========================
def render_search_results_base(df_res, show_meaning=False, show_kana=False, prob=0.0, karuta_mode=False, kimariji_len=None, show_jou=False, show_ge=False):
    """
    1件ずつHTMLに整形する基本関数。
    karuta_mode=False and show_kana=False の場合に「URL」列のリンクを適用します。
    """
    if df_res.empty:
        return "（該当なし）"
    html = ""
    
    kim_len_filter = int(kimariji_len) if kimariji_len not in (None,"None","") and karuta_mode else 0
    
    # リンクを適用するかどうかのフラグ
    # 学習用タブは karuta_mode=False, show_kana=False で呼び出される
    # 書道用タブは karuta_mode=False, show_kana=True/False で呼び出される
    # リンクを適用するのは学習用タブのみ、つまり karuta_mode=False かつ show_kana=False の場合
    apply_link = not karuta_mode and not show_kana
    
    for _, row in df_res.iterrows():
        upper = row.get('上の句','')
        lower = row.get('下の句','')
        upper_kana = row.get('上の句（ひらがな）', upper)
        lower_kana = row.get('下の句（ひらがな）', lower)
        url = row.get('URL', '') # URL列の値を取得
        
        # 競技かるたモードの表示制御
        display_upper = upper if not show_ge else ""  # 下の句だけ表示なら上の句は空
        display_lower = lower if not show_jou else ""  # 上の句だけ表示なら下の句は空

        html += f"<b>{row.get('No','')}</b>　{row.get('歌人','')}<br>"
        
        # 和歌にリンクを適用 (学習用タブのみ)
        poem_text = f"{display_upper}　{display_lower}"
        if apply_link and url:
            # URLが存在する場合、和歌全体にリンクを適用
            html += f'<a href="{url}" target="_blank" style="text-decoration: none; color: inherit;">{poem_text}</a><br>'
        else:
            html += f"{poem_text}<br>"
        
        # ひらがな/変体仮名表示 (書道用/学習用)
        if show_kana:
            if karuta_mode and '上の句決まり字数' in df_res.columns:
                # かるたモードでは決まり字を強調
                u_len = int(row.get('上の句決まり字数',0) or 0)
                # '下の句決まり字数'列がないため、上の句の決まり字長 u_len を下の句の強調にも流用（ただし下の句は「初句」の強調は不要なのでこれは不自然かも）
                # ここでは、元のコードに合わせて「決まり字」の強調として上の句の決まり字数を使用
                upper_html = f"<span style='color:red;font-weight:bold'>{upper_kana[:u_len]}</span>{upper_kana[u_len:]}" if u_len>0 else upper_kana
                lower_html = f"<span style='color:blue;font-weight:bold'>{lower_kana[:u_len]}</span>{lower_kana[u_len:]}" if u_len>0 else lower_kana
                
                # かるたの表示制御を反映
                upper_html = upper_html if not show_ge else ""
                lower_html = lower_html if not show_jou else ""
                html += f"ひらがな（決まり字強調）：{upper_html}　{lower_html}<br>"
            
            # 書道用タブの場合はひらがな/変体仮名を表示 (karuta_mode=False and show_kana=True)
            elif not karuta_mode and prob > 0: 
                html += f"ひらがな：{upper_kana}　{lower_kana}<br>"
                html += f"変体仮名：{random_hentaigana(upper_kana+'　'+lower_kana,hentaigana_map,prob)}<br>"

        if show_meaning and row.get('現代語訳',''):
            html += f'<i style="color: #0c4f0e;">{row.get("現代語訳","")}</i><br>' # <--- 修正箇所

        html += "<hr>"
    return html

# ==========================
# タブ関数 (学習用)
# ==========================
def safe_learning_mode_v3(df, text_kw, author_kw, number_kw, tags, show_meaning, prob=0.3):
    """学習用: OR検索対応、条件ごとに独立表示、現代語訳/変体仮名も出力"""
    tags = tags or []
    sections = []
    all_indices = set()
    
    # 学習用タブでは show_kana=False に固定する (render_search_results_base のリンク適用フラグに使用)
    show_kana_learning = False 
    
    # 検索条件とその表示名
    conditions = []
    if text_kw: conditions.append(("本文", text_kw, search_poems(df, text_kw_raw=text_kw)))
    if author_kw: conditions.append(("作者", author_kw, search_poems(df, author_kw_raw=author_kw)))
    if number_kw: conditions.append(("歌番号", ','.join(map(str, parse_numbers(number_kw))), search_poems(df, number_kw_raw=number_kw)))
    for t in tags: conditions.append(("タグ", t, search_poems(df, tags_raw=[t])))

    if not conditions:
        df_all = search_poems(df)
        # show_kana_learningを適用 (False)
        sections.append(f"■全件表示：{len(df_all)}首<br>{render_search_results_base(df_all, show_meaning=show_meaning, show_kana=show_kana_learning, prob=prob)}")
        all_indices.update(df_all.index)
    else:
        for type, kw_display, df_res in conditions:
            # show_kana_learningを適用 (False)
            sections.append(f"■{type}「{kw_display}」：{len(df_res)}首<br>{render_search_results_base(df_res, show_meaning=show_meaning, show_kana=show_kana_learning, prob=prob)}")
            all_indices.update(df_res.index)
    
    total_poems = len(all_indices)
    summary_html = f"【和歌をタップすると、外部の解説ページにリンクします】<br>"
    return summary_html + "<br>".join(sections)

# ==========================
# タブ関数 (書道用) (変更なし)
# ==========================
def safe_calligraphy_mode_v3(df, text_kw, author_kw, tags, show_kana, prob):
    """書道用: OR検索対応、条件ごとに独立表示、ひらがな/変体仮名表示に特化"""
    tags = tags or []
    sections = []
    all_indices = set()
    
    conditions = []
    if text_kw: conditions.append(("本文", text_kw, search_poems(df, text_kw_raw=text_kw)))
    if author_kw: conditions.append(("作者", author_kw, search_poems(df, author_kw_raw=author_kw)))
    for t in tags: conditions.append(("タグ", t, search_poems(df, tags_raw=[t])))

    if not conditions:
        df_all = search_poems(df)
        sections.append(f"■全件表示：{len(df_all)}首<br>{render_search_results_base(df_all, show_meaning=False, show_kana=show_kana, prob=prob)}")
        all_indices.update(df_all.index)
    else:
        for type, kw_display, df_res in conditions:
            sections.append(f"■{type}「{kw_display}」：{len(df_res)}首<br>{render_search_results_base(df_res, show_meaning=False, show_kana=show_kana, prob=prob)}")
            all_indices.update(df_res.index)
    
    total_poems = len(all_indices)
    summary_html = f"【和歌をタップすると、外部の解説ページにリンクします】<br>"
    return summary_html + "<br>".join(sections)

# ==========================
# タブ関数 (競技かるた用) (変更なし)
# ==========================
def safe_karuta_mode_v3(df, text_kw, author_kw, number_kw, show_kana, kimariji_len, show_jou, show_ge):
    """競技かるた用: OR検索対応、条件ごとに独立表示、決まり字フィルタ/強調/句表示制御"""
    kimariji_len_str = None if str(kimariji_len) in ("None","","None") else str(kimariji_len)
    kim_len_filter = int(kimariji_len_str) if kimariji_len_str else 0
    
    # かるた用は、検索条件ごとにフィルタリング・表示制御を行う必要があるため、
    # 各検索結果に対して個別に決まり字フィルタを適用する。
    
    sections = []
    all_indices = set()
    
    # 歌番号検索は、決まり字によるフィルタリング前に実施
    df_number_full = pd.DataFrame()
    if number_kw:
        df_number_full = search_poems(df, number_kw_raw=number_kw)
        
    def apply_karuta_filter(df_input):
        if kim_len_filter > 0 and '上の句決まり字数' in df_input.columns:
            return df_input[pd.to_numeric(df_input['上の句決まり字数'], errors='coerce').fillna(0) == kim_len_filter]
        return df_input
    
    conditions = []
    if text_kw: conditions.append(("本文", text_kw, search_poems(df, text_kw_raw=text_kw)))
    if author_kw: conditions.append(("作者", author_kw, search_poems(df, author_kw_raw=author_kw)))
    if number_kw: conditions.append(("歌番号", ','.join(map(str, parse_numbers(number_kw))), df_number_full))
    # 競技かるた用はタグUIがないため、タグ検索はスキップ

    if not conditions:
        df_all = search_poems(df)
        df_filtered = apply_karuta_filter(df_all)
        sections.append(f"■全件表示（決まり字フィルタ適用後）：{len(df_filtered)}首<br>{render_search_results_base(df_filtered, show_kana=show_kana, karuta_mode=True, kimariji_len=kim_len_filter, show_jou=show_jou, show_ge=show_ge)}")
        all_indices.update(df_filtered.index)
    else:
        for type, kw_display, df_res in conditions:
            df_filtered = apply_karuta_filter(df_res)
            
            # 歌番号でフィルタした場合、フィルタ情報も併記
            kimariji_info = f"（決まり字{kim_len_filter}字フィルタ適用）" if kim_len_filter > 0 else ""
            
            sections.append(f"■{type}「{kw_display}」{kimariji_info}：{len(df_filtered)}首<br>{render_search_results_base(df_filtered, show_kana=show_kana, karuta_mode=True, kimariji_len=kim_len_filter, show_jou=show_jou, show_ge=show_ge)}")
            all_indices.update(df_filtered.index)
    
    total_poems = len(all_indices)
    summary_html = f"【競技かるた用検索結果】<br>"
    
    if kim_len_filter > 0:
        summary_html += f"**※ 上の句決まり字数{kim_len_filter}字でフィルタリングしています**<br>"
        
    return summary_html + "<br>".join(sections)

# ==========================
# 安全ラッパー (UI連携用) (変更なし)
# ==========================
# グローバルなデータフレームをロード
df = load_csv("OguraHyakuninIshu.csv")

# 既存のラッパー関数は新しいv3関数を参照するように変更
def learning_update_safe(text_kw, author_kw, number_kw, tags, show_meaning):
    # safe_learning_mode_v3を呼び出す
    return safe_learning_mode_v3(df, text_kw, author_kw, number_kw, tags, show_meaning, prob=0.3)

def calligraphy_update_safe(calligraphy_text, calligraphy_author, calligraphy_tags, calligraphy_show_kana, prob_slider=0.4):
    return safe_calligraphy_mode_v3(df, calligraphy_text, calligraphy_author, calligraphy_tags, calligraphy_show_kana, prob=prob_slider)

def karuta_update_safe(text_kw, author_kw, number_kw, show_kana, kimariji_len, show_jou, show_ge):
    return safe_karuta_mode_v3(df, text_kw, author_kw, number_kw, show_kana, kimariji_len, show_jou, show_ge)


def clear_learning(): return "","","",[],False
def clear_calligraphy(): return "","",[],False,0.4
def clear_karuta(): return "","","",False,"None",False,False

# ==========================
# Gradio UI (変更なし)
# ==========================
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 百人一首検索アプリ
        - 学習・書道・競技かるたと、用途に応じて、ご利用いただけます
        - 現代語訳は、時代背景などを踏まえ、意訳しております。
        - 和歌を変体仮名へ変換できます。作品制作にお役立てください。
        """
    ) # タイトルを追記
    with gr.Tabs():
        # 学習用
        with gr.TabItem("学習用"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_kw = gr.Textbox(label="本文フリーワード")
                    author_kw = gr.Textbox(label="作者フリーワード")
                    number_kw = gr.Textbox(label="歌番号検索")
                    tags = gr.CheckboxGroup(label="ジャンルタグ",choices=['男性','女性','上皇','僧','春','夏','秋','冬','恋','別れ','旅'])
                    show_meaning = gr.Checkbox(label="現代語訳表示")
                    search_btn = gr.Button("検索")
                    clear_btn = gr.Button("クリア")
                with gr.Column(scale=30):
                    output_learning = gr.HTML(label="検索結果")
                    
            # search_btn.click
            search_btn.click(
                fn=learning_update_safe,
                inputs=[text_kw, author_kw, number_kw, tags, show_meaning],
                outputs=output_learning
            )
            # clear_btn.click
            clear_btn.click(fn=clear_learning,inputs=None,outputs=[text_kw,author_kw,number_kw,tags,show_meaning]) 
            
            # w.change 
            for w in [text_kw,author_kw,number_kw,tags,show_meaning]:
                w.change(
                    fn=learning_update_safe,
                    inputs=[text_kw,author_kw,number_kw,tags,show_meaning],
                    outputs=output_learning
                )

        # 書道用
        with gr.TabItem("書道用"):
            with gr.Row():
                with gr.Column(scale=1):
                    calligraphy_text = gr.Textbox(label="本文検索")
                    calligraphy_author = gr.Textbox(label="作者検索")
                    calligraphy_tags = gr.CheckboxGroup(label="ジャンルタグ",choices=['男性','女性','上皇','僧','春','夏','秋','冬','恋','別れ','旅'])
                    calligraphy_show_kana = gr.Checkbox(label="ひらがな/変体仮名表示")
                    prob_slider = gr.Slider(minimum=0.2,maximum=0.8,step=0.05,value=0.4,label="変換度合い（少なめ ←→ 多め）",interactive=True)
                    calligraphy_btn = gr.Button("検索")
                    calligraphy_random = gr.Button("ランダム変体仮名（再描画）")
                    calligraphy_clear = gr.Button("クリア")
                with gr.Column(scale=30):
                    output_calligraphy = gr.HTML(label="書道用結果")
                    
            # calligraphy_btn.click 
            calligraphy_btn.click(
                fn=calligraphy_update_safe,
                inputs=[calligraphy_text, calligraphy_author, calligraphy_tags, calligraphy_show_kana, prob_slider],
                outputs=output_calligraphy
            )
            
            # calligraphy_clear.click 
            calligraphy_clear.click(
                fn=clear_calligraphy,
                inputs=None,
                outputs=[calligraphy_text, calligraphy_author, calligraphy_tags, calligraphy_show_kana, prob_slider, output_calligraphy]
            )
            
            # calligraphy_random.click (再描画)
            calligraphy_random.click(
                fn=calligraphy_update_safe,
                inputs=[calligraphy_text, calligraphy_author, calligraphy_tags, calligraphy_show_kana, prob_slider],
                outputs=output_calligraphy
            )
            
            # w.change 
            for w in [calligraphy_text, calligraphy_author, calligraphy_tags, calligraphy_show_kana, prob_slider]:
                w.change(
                    fn=calligraphy_update_safe,
                    inputs=[calligraphy_text, calligraphy_author, calligraphy_tags, calligraphy_show_kana, prob_slider],
                    outputs=output_calligraphy
                )
            
        # 競技かるた用
        with gr.TabItem("競技かるた用"):
            with gr.Row():
                with gr.Column(scale=1):
                    karuta_text = gr.Textbox(label="本文検索")
                    karuta_author = gr.Textbox(label="作者検索")
                    karuta_number = gr.Textbox(label="歌番号検索")
                    karuta_kimariji_len = gr.Dropdown(label="上の句決まり字数検索",choices=["None","1","2","3","4","5","6"],value="None")
                    karuta_show_kana = gr.Checkbox(label="ひらがな（決まり字強調）表示", value=False)
                    karuta_show_jou = gr.Checkbox(label="上の句だけ表示", value=False)
                    karuta_show_ge = gr.Checkbox(label="下の句だけ表示", value=False)
                    karuta_btn = gr.Button("検索")
                    karuta_clear = gr.Button("クリア")
                with gr.Column(scale=30):
                    output_karuta = gr.HTML(label="競技かるた結果")
                    
                    # karuta_btn.click
                    karuta_btn.click(
                        fn=karuta_update_safe,
                        inputs=[karuta_text, karuta_author, karuta_number, karuta_show_kana, karuta_kimariji_len, karuta_show_jou, karuta_show_ge],
                        outputs=output_karuta
                    )

                    # karuta_clear.click 
                    karuta_clear.click(
                        fn=clear_karuta,
                        inputs=None,
                        outputs=[karuta_text, karuta_author, karuta_number, karuta_show_kana, karuta_kimariji_len, karuta_show_jou, karuta_show_ge, output_karuta]
                    )

                    # w.change
                    for w in [karuta_text, karuta_author, karuta_number, karuta_show_kana, karuta_kimariji_len, karuta_show_jou, karuta_show_ge]:
                        w.change(
                            fn=karuta_update_safe,
                            inputs=[karuta_text, karuta_author, karuta_number, karuta_show_kana, karuta_kimariji_len, karuta_show_jou, karuta_show_ge],
                            outputs=output_karuta
                        )

    # --- 参考サイトの追加 ---
    gr.HTML("<hr>")
    gr.Markdown(
        """
        ## 参考サイト
        - <a href="https://oumijingu.org/pages/130/" target="_blank">近江神宮-小倉百人一首一覧</a>
        - <a href="https://ogurasansou.jp.net/columns_category/hyakunin/" target="_blank">小倉山荘-ちょっと差がつく『百人一首講座』</a>
        - <a href="https://hyakuninisshu.sakura.ne.jp/list.html" target="_blank">時雨の百人一首</a>
        - <a href="https://www.samac.jp/search/poems_list.php" target="_blank">【嵯峨嵐山文華館】小倉百人一首の全首を見る</a>
        """
    )
    # ----------------------

demo.launch(share=False)