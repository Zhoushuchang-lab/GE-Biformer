import os
import xml.sax.saxutils as saxutils


def escape(s):
    return saxutils.escape(s)


def make_cell(cell_id, parent, style, value, x, y, w, h, vertex="1", visible="1"):
    geo = f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
    return f'<mxCell id="{cell_id}" value="{escape(value)}" style="{escape(style)}" vertex="{vertex}" visible="{visible}" parent="{parent}">{geo}</mxCell>'


def make_edge(cell_id, parent, style, source, target, value="", visible="1"):
    geo = '<mxGeometry relative="1" as="geometry" />'
    return f'<mxCell id="{cell_id}" value="{escape(value)}" style="{escape(style)}" edge="1" visible="{visible}" parent="{parent}" source="{source}" target="{target}">{geo}</mxCell>'


def make_edge_with_points(cell_id, parent, style, source, target, value, points, visible="1"):
    pts_xml = " ".join(f'<mxPoint x="{px}" y="{py}" as="sourcePoint" />' if i == 0 else
                       f'<mxPoint x="{px}" y="{py}" as="targetPoint" />' if i == len(points) - 1 else
                       f'<mxPoint x="{px}" y="{py}" />'
                       for i, (px, py) in enumerate(points))
    geo = f'<mxGeometry relative="1" as="geometry"><Array as="points">{pts_xml}</Array></mxGeometry>'
    return f'<mxCell id="{cell_id}" value="{escape(value)}" style="{escape(style)}" edge="1" visible="{visible}" parent="{parent}" source="{source}" target="{target}">{geo}</mxCell>'


# --- color gradients for heatmap ---
def heatmap_color(row, col, max_row, max_col):
    intensity = (row * max_col + col) / (max_row * max_col - 1)
    r = int(255 - 100 * intensity)
    g = int(200 - 140 * intensity)
    b = int(255 - 100 * intensity)
    return f"#{r:02X}{g:02X}{b:02X}"


def build_xml():
    cells = []

    # ---- root cells ----
    cells.append('<mxCell id="0" />')
    cells.append('<mxCell id="1" parent="0" />')

    # =====================================================================
    # STEP 1: 数据输入  (x ≈ 30..290)
    # =====================================================================
    sx1, sy_label, sy_content, sy_note = 30, 20, 100, 240
    step1_col_w = 110
    gap = 16

    # Step 1 label
    cells.append(make_cell("s1_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;fontStyle=1;fontColor=#2C3E50;",
        "<b>Step 1: 数据维度</b>", sx1, sy_label, 250, 28))

    # SNP Matrix box
    cells.append(make_cell("snp_box", "1",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#D6E8FF;strokeColor=#4A90D9;strokeWidth=2;fontSize=12;fontColor=#2C3E50;",
        "Input SNP Matrix&lt;br&gt;(N × S)", sx1, sy_content, step1_col_w, 60))

    # Env Matrix box
    cells.append(make_cell("env_box", "1",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#D5F5E3;strokeColor=#27AE60;strokeWidth=2;fontSize=12;fontColor=#2C3E50;",
        "Input Env Matrix&lt;br&gt;(N × E)", sx1 + step1_col_w + gap, sy_content, step1_col_w, 60))

    # Note below
    cells.append(make_cell("s1_note", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=10;fontColor=#7F8C8D;",
        "N = hybrids, S = SNPs, E = env vars", sx1, sy_content + 75, 250, 24))

    step1_right = sx1 + 250  # right edge of step 1 content

    # =====================================================================
    # STEP 2: 可学习聚类中心  (x ≈ 310..500)
    # =====================================================================
    sx2, sy_label2 = 310, sy_label
    sy_proto_area = sy_content - 10

    cells.append(make_cell("s2_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;fontStyle=1;fontColor=#2C3E50;",
        "<b>Step 2: 可学习聚类中心</b>", sx2, sy_label2, 220, 28))

    # Group box for prototypes
    gb_x, gb_y, gb_w, gb_h = sx2, sy_proto_area, 190, 210
    cells.append(make_cell("s2_group", "1",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#BDC3C7;strokeWidth=1;dashed=1;dashPattern=5 5;",
        "", gb_x, gb_y, gb_w, gb_h))

    # Label inside group
    cells.append(make_cell("s2_grp_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=11;fontStyle=1;fontColor=#555555;",
        "Learnable Token Prototypes", gb_x, gb_y + 5, gb_w, 22))

    # 8 SNP prototypes (circles) — orange
    snp_proto_y = gb_y + 35
    snp_circles = []
    proto_r = 14
    proto_gap = 6
    for i in range(8):
        cx = gb_x + 10 + i * (proto_r + proto_gap)
        cy = snp_proto_y
        cid = f"s2_snp_p{i+1}"
        cells.append(make_cell(cid, "1",
            f"ellipse;whiteSpace=wrap;html=1;fillColor=#FFE0B2;strokeColor=#E67E22;strokeWidth=1.5;fontSize=7;fontColor=#A04000;",
            f"P{i+1}&lt;sup&gt;s&lt;/sup&gt;", cx - proto_r, cy - proto_r, 2 * proto_r, 2 * proto_r))
        snp_circles.append(cid)

    # SNP label below circles
    cells.append(make_cell("s2_snp_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=9;fontColor=#E67E22;fontStyle=2;",
        "dim = S", gb_x, snp_proto_y + 28, gb_w, 16))

    # 4 Env prototypes (circles) — teal
    env_proto_y = snp_proto_y + 55
    env_circles = []
    for i in range(4):
        cx = gb_x + 20 + i * (proto_r + proto_gap + 6)
        cy = env_proto_y
        cid = f"s2_env_p{i+1}"
        cells.append(make_cell(cid, "1",
            f"ellipse;whiteSpace=wrap;html=1;fillColor=#B2EBF2;strokeColor=#00838F;strokeWidth=1.5;fontSize=7;fontColor=#004D40;",
            f"P{i+1}&lt;sup&gt;e&lt;/sup&gt;", cx - proto_r, cy - proto_r, 2 * proto_r, 2 * proto_r))
        env_circles.append(cid)

    # Env label below circles
    cells.append(make_cell("s2_env_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=9;fontColor=#00838F;fontStyle=2;",
        "dim = E", gb_x, env_proto_y + 28, gb_w, 16))

    step2_right = sx2 + gb_w

    # =====================================================================
    # STEP 3: 相似度计算与软分配  (x ≈ 540..760)
    # =====================================================================
    sx3 = 540
    cells.append(make_cell("s3_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;fontStyle=1;fontColor=#2C3E50;",
        "<b>Step 3: 相似度计算 &amp; 软分配</b>", sx3, sy_label, 220, 28))

    # Formula box
    formula1_y = sy_content - 10
    cells.append(make_cell("formula1", "1",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF9E7;strokeColor=#F1C40F;strokeWidth=2;fontSize=11;fontFamily=Courier New;fontColor=#7D6608;",
        "Sⁿ = SNP · Pˢᵀ / τ", sx3, formula1_y, 200, 36))

    # Heatmap grid (N × K): 6 rows × 8 cols
    heat_x, heat_y = sx3 + 10, formula1_y + 55
    cell_w, cell_h = 22, 17
    n_rows, n_cols = 6, 8
    heat_cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            cid = f"heat_{r}_{c}"
            fill = heatmap_color(r, c, n_rows, n_cols)
            cells.append(make_cell(cid, "1",
                f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#999999;strokeWidth=0.5;",
                "", heat_x + c * cell_w, heat_y + r * cell_h, cell_w, cell_h))
            heat_cells.append(cid)

    # Heatmap label
    cells.append(make_cell("heat_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=10;fontColor=#7F8C8D;fontStyle=2;",
        "Assignment Matrix (N × K)", heat_x, heat_y + n_rows * cell_h + 4, n_cols * cell_w, 18))

    # Arrow from formula to heatmap
    cells.append(make_edge("arrow_f2h", "1",
        "endArrow=classic;html=1;rounded=0;strokeColor=#F1C40F;strokeWidth=1.5;",
        "formula1", "heat_0_0",
        ""))

    step3_right = sx3 + 220

    # =====================================================================
    # STEP 4: Token生成  (x ≈ 800..1000)
    # =====================================================================
    sx4 = 800
    cells.append(make_cell("s4_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;fontStyle=1;fontColor=#2C3E50;",
        "<b>Step 4: Token 生成</b>", sx4, sy_label, 220, 28))

    # Formula box
    formula2_y = sy_content - 10
    cells.append(make_cell("formula2", "1",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF9E7;strokeColor=#F1C40F;strokeWidth=2;fontSize=10;fontFamily=Courier New;fontColor=#7D6608;",
        "Tⱼˢⁿᵖ = Σᵢ(αᵢⱼ · SNPᵢ) / Σᵢαᵢⱼ", sx4, formula2_y, 200, 36))

    # SNP Tokens (8 rectangles, red)
    snp_token_y = formula2_y + 55
    token_w, token_h = 18, 34
    token_gap = 4
    snp_token_ids = []
    for i in range(8):
        tx = sx4 + 8 + i * (token_w + token_gap)
        cid = f"token_snp_{i+1}"
        cells.append(make_cell(cid, "1",
            f"rounded=1;whiteSpace=wrap;html=1;fillColor=#FADBD8;strokeColor=#E74C3C;strokeWidth=1.5;fontSize=6;fontColor=#922B21;",
            f"T{i+1}&lt;sup&gt;SNP&lt;/sup&gt;", tx, snp_token_y, token_w, token_h))
        snp_token_ids.append(cid)

    # SNP Token label
    cells.append(make_cell("snp_token_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=9;fontColor=#E74C3C;fontStyle=2;",
        "K_snp × F", sx4, snp_token_y + token_h + 4, 200, 14))

    # Env Tokens (4 rectangles, purple)
    env_token_y = snp_token_y + token_h + 25
    env_token_ids = []
    for i in range(4):
        tx = sx4 + 30 + i * (token_w + token_gap + 6)
        cid = f"token_env_{i+1}"
        cells.append(make_cell(cid, "1",
            f"rounded=1;whiteSpace=wrap;html=1;fillColor=#E8DAEF;strokeColor=#8E44AD;strokeWidth=1.5;fontSize=6;fontColor=#6C3483;",
            f"T{i+1}&lt;sup&gt;ENV&lt;/sup&gt;", tx, env_token_y, token_w, token_h))
        env_token_ids.append(cid)

    # Env Token label
    cells.append(make_cell("env_token_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=9;fontColor=#8E44AD;fontStyle=2;",
        "K_env × F", sx4, env_token_y + token_h + 4, 200, 14))

    step4_right = sx4 + 220

    # =====================================================================
    # STEP 5: Token层次自注意力交互  (x ≈ 1050..1280)
    # =====================================================================
    sx5 = 1050
    cells.append(make_cell("s5_label", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;fontStyle=1;fontColor=#2C3E50;",
        "<b>Step 5: Token 自注意力交互</b>", sx5, sy_label, 240, 28))

    # Cross-Modal Multi-Head Attention block
    attn_x, attn_y, attn_w, attn_h = sx5, sy_content - 15, 230, 220
    cells.append(make_cell("attn_block", "1",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#E8DAEF;strokeColor=#8E44AD;strokeWidth=2;fontSize=12;fontColor=#6C3483;fontStyle=1;",
        "Cross-Modal&lt;br&gt;Multi-Head Attention", attn_x + 20, attn_y + 15, attn_w - 40, 35))

    # Q/K/V flow arrows inside attention block
    # SNP Tokens → Q
    cells.append(make_cell("qkv_q", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=10;fontColor=#922B21;fontStyle=1;",
        "Q", attn_x + 15, attn_y + 70, 24, 18))

    cells.append(make_cell("qkv_k", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=10;fontColor=#6C3483;fontStyle=1;",
        "K", attn_x + 55, attn_y + 70, 24, 18))

    cells.append(make_cell("qkv_v", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=10;fontColor=#6C3483;fontStyle=1;",
        "V", attn_x + 95, attn_y + 70, 24, 18))

    # Arrow from Q to next layer
    cells.append(make_edge("qkv_q_arrow", "1",
        "endArrow=classic;html=1;rounded=0;strokeColor=#E74C3C;strokeWidth=1;",
        "qkv_q", "qkv_k", ""))

    cells.append(make_edge("qkv_k_arrow", "1",
        "endArrow=classic;html=1;rounded=0;strokeColor=#8E44AD;strokeWidth=1;",
        "qkv_k", "qkv_v", ""))

    # Attention formula inside
    cells.append(make_cell("attn_formula", "1",
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=9;fontColor=#6C3483;fontStyle=2;",
        "Attention(Q,K,V) = softmax(QKᵀ/√d)V", attn_x + 10, attn_y + 95, attn_w - 20, 18))

    # Fused representation output
    cells.append(make_cell("fused_box", "1",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#D5F5E3;strokeColor=#1ABC9C;strokeWidth=2;fontSize=11;fontColor=#0E6655;fontStyle=1;",
        "Fused Token&lt;br&gt;Representation&lt;br&gt;(dim = F)", attn_x + 35, attn_y + 130, attn_w - 70, 58))

    # Arrow from QKV to fused
    cells.append(make_edge("qkv2fused", "1",
        "endArrow=classic;html=1;rounded=0;strokeColor=#8E44AD;strokeWidth=1.5;",
        "qkv_v", "fused_box", ""))

    step5_right = sx5 + attn_w

    # =====================================================================
    # CONNECTION ARROWS BETWEEN STEPS
    # =====================================================================
    arrow_style = "endArrow=classic;html=1;rounded=0;strokeColor=#7F8C8D;strokeWidth=2.5;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;"

    # Step 1 → Step 2: from env_box right to s2_group left
    cells.append(make_edge("arrow_1_2", "1",
        arrow_style,
        "env_box", "s2_group", ""))

    # Step 2 → Step 3: from s2_group right to formula1 left
    cells.append(make_edge("arrow_2_3", "1",
        arrow_style,
        "s2_group", "formula1", ""))

    # Step 3 → Step 4: from heatmap right area to formula2 left
    cells.append(make_edge("arrow_3_4", "1",
        arrow_style,
        "heat_label", "formula2", ""))

    # Step 4 → Step 5: from env token area to attn_block
    cells.append(make_edge("arrow_4_5", "1",
        arrow_style,
        "token_env_1", "attn_block", ""))

    # Additional internal arrows for clarity
    # Step 1 → Step 3 conceptual link (scatter arrow from SNP to formula)
    cells.append(make_edge("data2formula", "1",
        "endArrow=classic;html=1;rounded=0;strokeColor=#BDC3C7;strokeWidth=1;dashed=1;dashPattern=3 3;",
        "snp_box", "formula1", ""))

    cells.append(make_edge("proto2formula", "1",
        "endArrow=classic;html=1;rounded=0;strokeColor=#BDC3C7;strokeWidth=1;dashed=1;dashPattern=3 3;",
        "s2_snp_p1", "formula1", ""))

    # =====================================================================
    # FINAL CANVAS SIZE: 1300 × 900
    # =====================================================================
    canvas_w = 1330
    canvas_h = 900

    # Assemble
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<mxfile host="GEBiformer" modified="2026-05-26T00:00:00.000Z" agent="Python" version="21.0.0" type="device">',
        f'  <diagram name="Tokenization Process" id="diagram_1">',
        f'    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{canvas_w}" pageHeight="{canvas_h}" math="0" shadow="0">',
        f'      <root>',
    ]

    for c in cells:
        xml_parts.append(f'        {c}')

    xml_parts.extend([
        f'      </root>',
        f'    </mxGraphModel>',
        f'  </diagram>',
        f'</mxfile>',
        '',
    ])

    return "\n".join(xml_parts)


def main():
    output_dir = r"d:\doc\GEBiformer\results\figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tokenization_process.drawio")

    xml_content = build_xml()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    file_size = os.path.getsize(output_path)
    print(f"DrawIO file written to: {output_path}")
    print(f"File size: {file_size} bytes")

    # Quick validation
    assert "mxfile" in xml_content, "Missing mxfile"
    assert "diagram" in xml_content, "Missing diagram"
    assert "mxGraphModel" in xml_content, "Missing mxGraphModel"
    print("XML validation: PASSED (contains mxfile, diagram, mxGraphModel)")


if __name__ == "__main__":
    main()
