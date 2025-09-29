import gradio as gr
import pandas as pd
import re

# ===== データ読み込み =====
df = pd.read_csv("OguraHyakuninIshu.csv")

# ===== キーワード柔軟分割 =====
def split_keywords(input_text):
    """
    入力文字列をカンマ、全角空白、半角空白、読点「、」で分割してリスト化
    """
    if not input_text:
        return []
    parts = re.split(r'[,\s　、]+', input_text)
    return [p.strip() for p in parts if p.strip()]

# ===== 出力整形関数 =====
def format_output(results, show_upper=True, show_upper_hira=False, show_lower=True, show_lower_hira=False, output_kimariji=False):
    outputs = []
    for res in results:
        if isinstance(res, dict):  # 歌リストのテキスト化
            display_cols = ["No", "歌人"]
            if show_upper:
                display_cols.append("上の句（ひらがな）" if show_upper_hira else "上の句")
            if show_lower:
                display_cols.append("下の句（ひらがな）" if show_lower_hira else "下の句")

            lines = []
            lines.append(res.get("header", ""))
            lines.append("\t".join(display_cols))
            for row in res["rows"]:
                line_parts = []
                for col in display_cols:
                    text = str(row[col])
            
                    # ===== 競技かるた用かつひらがな表示のときだけスラッシュ =====
                    if output_kimariji:
                        if show_upper_hira and col == "上の句（ひらがな）":
                            k = row["上の句決まり字数"]
                            if k > 0 and k < len(text):
                                text = text[:k] + '/' + text[k:]
                        elif show_lower_hira and col == "下の句（ひらがな）":
                            k = row["下の句決まり字数"]
                            if k > 0 and k < len(text):
                                text = text[:k] + '/' + text[k:]
                    line_parts.append(text)
                line = "\t".join(line_parts)
                lines.append(line)
            outputs.append("\n".join(lines))

        elif isinstance(res, str):
            outputs.append(res)

    return "\n\n".join(outputs)

# ===== 決まり字検索 =====
def output_kimariji(df, mode="both", upper_list=None, lower_list=None, show_upper_hira=False, show_lower_hira=False):
    outputs = []
    mode = mode.lower()

    # 上の句
    if mode in ("upper", "both") and upper_list:
        for k in upper_list:
            if k == "":
                continue
            k = int(k)
            subset = df[df["上の句決まり字数"] == k]
            if not subset.empty:
                rows = subset.to_dict("records")
                outputs.append({
                    "header": f"\n上の句{k}字決まりの歌：{len(subset)}首",
                    "rows": rows
                })

    # 下の句
    if mode in ("lower", "both") and lower_list:
        for k in lower_list:
            if k == "":
                continue
            k = int(k)
            subset = df[df["下の句決まり字数"] == k]
            if not subset.empty:
                rows = subset.to_dict("records")
                outputs.append({
                    "header": f"\n下の句{k}字決まりの歌：{len(subset)}首",
                    "rows": rows
                })

    return outputs

# ===== 検索処理 =====
def search_poems(text, author, numbers, tags, use_kimariji,
                 kimariji_mode, upper_num, lower_num):

    result = df.copy()
    outputs = []

    # 競技かるた検索
    if use_kimariji:
        upper_list = [upper_num] if upper_num and upper_num != "" else []
        lower_list = [lower_num] if lower_num and lower_num != "" else []
        mode_map = {"上の句": "upper", "下の句": "lower", "両方": "both"}
        mode = mode_map.get(kimariji_mode, "both")
        kimariji_res = output_kimariji(
            result,
            mode=mode,
            upper_list=upper_list,
            lower_list=lower_list
        )
        outputs.extend(kimariji_res)

    # 歌番号検索
    if numbers:
        numbers = numbers.replace(" ", "")
        nums = []
        for part in numbers.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                nums.extend(range(start, end + 1))
            else:
                nums.append(int(part))
        subset = result[result["No"].isin(nums)]
        if not subset.empty:
            outputs.append({
                "header": f"「歌番号 {numbers}」の歌：{len(subset)}首",
                "rows": subset.to_dict("records")
            })
        else:
            outputs.append(f"「歌番号 {numbers}」の歌は見つかりませんでした。")

    # 本文検索
    if text:
        keywords = split_keywords(text)
        for kw in keywords:
            mask = (
                result["上の句"].str.contains(kw, na=False) |
                result["下の句"].str.contains(kw, na=False) |
                result["上の句（ひらがな）"].str.contains(kw, na=False) |
                result["下の句（ひらがな）"].str.contains(kw, na=False)
            )
            subset = result[mask]
            if not subset.empty:
                outputs.append({
                    "header": f"「{kw}」を含む歌：{len(subset)}首",
                    "rows": subset.to_dict("records")
                })
            else:
                outputs.append(f"「{kw}」を含む歌は見つかりませんでした。")

    # 作者検索
    if author:
        authors = split_keywords(author)
        for a in authors:
            if a == "上皇":
                subset = result[result["上皇"] == 1]
                if not subset.empty:
                    outputs.append({
                        "header": f"「{a}」に分類される歌：{len(subset)}首",
                        "rows": subset.to_dict("records")
                    })
                else:
                    outputs.append(f"「{a}」に分類される歌は見つかりませんでした。")
            elif a in ["僧侶", "僧", "坊主", "お坊さん", "坊さん"]:
                subset = result[result["僧"] == 1]
                if not subset.empty:
                    outputs.append({
                        "header": f"「{a}」に分類される歌：{len(subset)}首",
                        "rows": subset.to_dict("records")
                    })
                else:
                    outputs.append(f"「{a}」に分類される歌は見つかりませんでした。")
            else:
                subset = result[
                    result["歌人"].str.contains(a, na=False) |
                    result["歌人（ひらがな）"].str.contains(a, na=False)
                ]
                if not subset.empty:
                    outputs.append({
                        "header": f"「{a}」が詠んだ歌：{len(subset)}首",
                        "rows": subset.to_dict("records")
                    })
                else:
                    outputs.append(f"「{a}」が詠んだ歌は見つかりませんでした。")

    # タグ検索
    if tags:
        for t in tags:
            if t in result.columns:
                subset = result[result[t] == 1]
                if not subset.empty:
                    outputs.append({
                        "header": f"「{t}」ジャンルの歌：{len(subset)}首",
                        "rows": subset.to_dict("records")
                    })
                else:
                    outputs.append(f"「{t}」ジャンルの歌は見つかりませんでした。")

    # 全件表示
    if not text and not author and not tags and not numbers and not use_kimariji:
        outputs.append({
            "header": "全件表示",
            "rows": result.to_dict("records")
        })

    return outputs

# ===== Gradio UI =====
with gr.Blocks(css="""
#search-btn {background-color: #FFB7C5 !important; color:white !important; border:none;}
.pink-checkbox input[type=checkbox]:checked {background-color:#FFB7C5 !important; border-color:#FFB7C5 !important;}
.pink-radio input[type=radio]:checked {background-color:#FFB7C5 !important; border-color:#FFB7C5 !important;}
/* スマホで検索窓を縦並びに */
@media (max-width: 768px) {
    .input-row {
        flex-direction: column !important;
    }
}
""") as demo:

    gr.Markdown(
        """
        # 百人一首検索
        
        **■使い方**  
        1. 歌番号検索は連番なら「1-10」のように、任意なら「3,5,7」のように入力してください。
        2. 「恋」など、ジャンル検索もできます。
        3. 上の句・下の句・ひらがな表示を変更できます。  
        4. 競技かるた用検索もできます。決まり字の数ごとに表示します。  
        """
    )

    with gr.Row(elem_classes="input-row"):
        text_input = gr.Textbox(label="本文検索", placeholder="例：春,秋")
        author_input = gr.Textbox(label="作者検索", placeholder="例：紫式部,清少納言")
        number_input = gr.Textbox(label="歌番号検索", placeholder="例：1-10,15,20")

    tag_input = gr.CheckboxGroup(["男性","女性","春","夏","秋","冬","恋","旅","別れ"], label="ジャンルタグ", elem_classes="pink-checkbox")

    with gr.Row():
        show_upper_check = gr.Checkbox(label="上の句を表示", value=True, elem_classes="pink-checkbox")
        show_lower_check = gr.Checkbox(label="下の句を表示", value=True, elem_classes="pink-checkbox")
        show_upper_hira = gr.Checkbox(label="上の句をひらがな表示", value=False, elem_classes="pink-checkbox")
        show_lower_hira = gr.Checkbox(label="下の句をひらがな表示", value=False, elem_classes="pink-checkbox")

    # 常に表示されるチェックボックス
    use_kimariji = gr.Checkbox(label="競技かるた用検索を有効にする", value=False, elem_classes="pink-checkbox")
    
    # チェック時のみ表示される設定グループ
    with gr.Group(visible=False) as kimariji_group:
        kimariji_mode = gr.Radio(["上の句","下の句","両方"], value="両方", label="表示対象", elem_classes="pink-radio")
        upper_num_dropdown = gr.Dropdown([""] + [str(i) for i in range(1, 7)],
                                         label="上の句の決まり字数", value="")
        lower_num_dropdown = gr.Dropdown([""] + [str(i) for i in range(1, 8)],
                                         label="下の句の決まり字数", value="")

    
    # チェックが切り替わったらグループの表示を切り替える
    def toggle_kimariji_group(checked):
        return gr.update(visible=checked)
    
    use_kimariji.change(toggle_kimariji_group, inputs=use_kimariji, outputs=kimariji_group)

        
    # チェックで表示切替
    def toggle_kimariji_group(is_checked):
        return gr.update(visible=is_checked)
    
    use_kimariji.change(toggle_kimariji_group, inputs=use_kimariji, outputs=kimariji_group)


    submit_btn = gr.Button("検索", elem_id="search-btn")
    clear_btn = gr.Button("クリア")

    output_box = gr.Textbox(label="検索結果", lines=30)

    search_cache = {"results": []}

    # 検索処理
    def on_search(text, author, numbers, tags, use_kimariji_chk,
                  kimariji_mode, upper_num, lower_num,
                  show_upper_chk, show_upper_hira_chk,
                  show_lower_chk, show_lower_hira_chk):
    
        # 検索
        res = search_poems(
            text, author, numbers, tags,
            use_kimariji=use_kimariji_chk,
            kimariji_mode=kimariji_mode,
            upper_num=upper_num,
            lower_num=lower_num
        )
    
        # 検索結果をキャッシュに保存
        search_cache["results"] = res
    
        # 出力整形
        return format_output(res,
                             show_upper=show_upper_chk,
                             show_upper_hira=show_upper_hira_chk,
                             show_lower=show_lower_chk,
                             show_lower_hira=show_lower_hira_chk,
                             output_kimariji=use_kimariji_chk)

    submit_btn.click(on_search,
                     inputs=[text_input, author_input, number_input, tag_input,
                             use_kimariji, kimariji_mode, upper_num_dropdown, lower_num_dropdown,
                             show_upper_check, show_upper_hira, show_lower_check, show_lower_hira],
                     outputs=output_box)

    def on_clear():
        search_cache["results"] = []
        return ""
    clear_btn.click(on_clear, inputs=[], outputs=output_box)

    # リアルタイム表示切替
    def update_display(show_upper, show_upper_hira, show_lower, show_lower_hira):
        return format_output(search_cache["results"],
                             show_upper=show_upper,
                             show_upper_hira=show_upper_hira,
                             show_lower=show_lower,
                             show_lower_hira=show_lower_hira)

    for cb in [show_upper_check, show_upper_hira, show_lower_check, show_lower_hira]:
        cb.change(update_display,
                  inputs=[show_upper_check, show_upper_hira, show_lower_check, show_lower_hira],
                  outputs=output_box)

demo.launch()
