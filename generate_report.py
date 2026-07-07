import os
import json
import math
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, String, Group
from reportlab.pdfgen import canvas

# Paleta de Cores
PRIMARY = HexColor("#1a2b4c")      # Azul Marinho Escuro (Confiança, Acadêmico)
SECONDARY = HexColor("#cda250")    # Dourado Escuro (Qualidade, Destaque)
DARK_TEXT = HexColor("#2c3e50")    # Cinza Azulado Escuro
LIGHT_BG = HexColor("#f8f9fa")     # Fundo muito claro
BORDER_COLOR = HexColor("#dcdde1") # Cinza claro para bordas
ERROR_COLOR = HexColor("#c62828")  # Vermelho para linhas de referência

class NumberedCanvas(canvas.Canvas):
    """Canvas de duas passagens para gerar cabeçalhos e numeração de página dinâmica (Página X de Y)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Pular cabeçalho e rodapé na capa (Página 1)
        if self._pageNumber == 1:
            # Desenhar faixa decorativa na capa
            self.setFillColor(PRIMARY)
            self.rect(0, 780, 595.27, 62, fill=True, stroke=False)
            self.setFillColor(SECONDARY)
            self.rect(0, 775, 595.27, 5, fill=True, stroke=False)
            self.restoreState()
            return
            
        # Desenhar Cabeçalho (Páginas 2 em diante)
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(PRIMARY)
        self.drawString(54, 800, "Predição de Notas de Estudantes - Análise de Gráficos e Visualizações")
        self.setFont("Helvetica", 8)
        self.setFillColor(DARK_TEXT)
        self.drawRightString(541.27, 800, "Inteligência Artificial II | Unifucamp")
        
        # Linha do Cabeçalho
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(54, 792, 541.27, 792)
        
        # Desenhar Rodapé
        self.line(54, 55, 541.27, 55)
        self.setFont("Helvetica", 8)
        self.drawString(54, 40, "Lucas Firmino Rodrigues e Carlos Eduardo Mendes")
        page_text = f"Página {self._pageNumber} de {page_count}"
        self.drawRightString(541.27, 40, page_text)
        
        self.restoreState()

# ==================== FUNÇÕES DE DESENHO VETORIAL ====================

def draw_histogram_flowable(labels, counts, title, fill_color=PRIMARY):
    """Desenha um gráfico de barras (histograma) no ReportLab."""
    d = Drawing(480, 160)
    
    # Fundo do gráfico
    d.add(Rect(0, 0, 480, 160, fillColor=colors.white, strokeColor=BORDER_COLOR, strokeWidth=0.5))
    
    margin_left = 35
    margin_bottom = 25
    margin_top = 22
    margin_right = 15
    
    plot_w = 480 - margin_left - margin_right
    plot_h = 160 - margin_bottom - margin_top
    
    max_count = max(counts) if counts else 1
    y_scale = plot_h / max_count
    
    # Título do Gráfico
    d.add(String(240, 145, title, textAnchor='middle', fontName='Helvetica-Bold', fontSize=9, fillColor=PRIMARY))
    
    # Desenhar eixos
    d.add(Line(margin_left, margin_bottom, margin_left + plot_w, margin_bottom, strokeColor=DARK_TEXT, strokeWidth=1))
    d.add(Line(margin_left, margin_bottom, margin_left, margin_bottom + plot_h, strokeColor=DARK_TEXT, strokeWidth=1))
    
    num_bars = len(counts)
    bar_gap = 4
    total_gaps_w = bar_gap * (num_bars + 1)
    bar_w = (plot_w - total_gaps_w) / num_bars
    
    for i, count in enumerate(counts):
        bar_h = count * y_scale
        bx = margin_left + bar_gap + i * (bar_w + bar_gap)
        by = margin_bottom
        
        d.add(Rect(bx, by, bar_w, bar_h, fillColor=fill_color, strokeColor=None))
        
        if bar_w > 12 and bar_h > 10:
            d.add(String(bx + bar_w/2, by + bar_h + 3, str(count), textAnchor='middle', fontName='Helvetica', fontSize=7, fillColor=DARK_TEXT))
            
        label = labels[i]
        d.add(String(bx + bar_w/2, margin_bottom - 12, label, textAnchor='middle', fontName='Helvetica', fontSize=7, fillColor=DARK_TEXT))
        
    ticks = 4
    for t in range(ticks + 1):
        val = int(max_count * (t / ticks))
        ty = margin_bottom + (t / ticks) * plot_h
        d.add(Line(margin_left - 3, ty, margin_left, ty, strokeColor=DARK_TEXT, strokeWidth=0.5))
        d.add(String(margin_left - 6, ty - 3, str(val), textAnchor='end', fontName='Helvetica', fontSize=7, fillColor=DARK_TEXT))
        
    return d

def draw_heatmap_flowable(corr_matrix, labels, title):
    """Desenha uma matriz de correlação em formato heatmap no ReportLab."""
    d = Drawing(480, 220)
    
    # Fundo
    d.add(Rect(0, 0, 480, 220, fillColor=colors.white, strokeColor=BORDER_COLOR, strokeWidth=0.5))
    
    margin_left = 80
    margin_bottom = 60
    margin_top = 22
    margin_right = 15
    
    plot_w = 480 - margin_left - margin_right
    plot_h = 220 - margin_bottom - margin_top
    
    n = len(labels)
    cell_w = plot_w / n
    cell_h = plot_h / n
    
    # Título
    d.add(String(240, 205, title, textAnchor='middle', fontName='Helvetica-Bold', fontSize=9, fillColor=PRIMARY))
    
    def get_corr_color(val):
        if val >= 0:
            f = val
            r = int(255 * (1 - f * 0.8))
            g = int(255 * (1 - f * 0.6))
            b = 255
        else:
            f = abs(val)
            r = 255
            g = int(255 * (1 - f * 0.6))
            b = int(255 * (1 - f * 0.8))
        return HexColor(f"#{r:02x}{g:02x}{b:02x}")
        
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i][j]
            cx = margin_left + j * cell_w
            cy = margin_bottom + (n - 1 - i) * cell_h
            
            cell_color = get_corr_color(val)
            d.add(Rect(cx, cy, cell_w, cell_h, fillColor=cell_color, strokeColor=HexColor("#e0e0e0"), strokeWidth=0.5))
            
            if cell_w > 18:
                txt_color = colors.white if abs(val) > 0.5 else DARK_TEXT
                d.add(String(cx + cell_w/2, cy + cell_h/2 - 3, f"{val:.2f}", textAnchor='middle', fontName='Helvetica-Bold', fontSize=6, fillColor=txt_color))
                
        d.add(String(margin_left - 6, margin_bottom + (n - 1 - i) * cell_h + cell_h/2 - 3, labels[i], textAnchor='end', fontName='Helvetica', fontSize=7, fillColor=DARK_TEXT))
        
        g = Group()
        g.translate(margin_left + i * cell_w + cell_w/2, margin_bottom - 6)
        g.rotate(-45)
        g.add(String(0, 0, labels[i], textAnchor='end', fontName='Helvetica', fontSize=7, fillColor=DARK_TEXT))
        d.add(g)
        
    return d

def draw_scatter_flowable(x, y, x_label, y_label, title, is_residual=False, width=225, height=150):
    """Desenha um gráfico de dispersão para o PDF."""
    d = Drawing(width, height)
    
    d.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=BORDER_COLOR, strokeWidth=0.5))
    
    margin_left = 30
    margin_bottom = 25
    margin_top = 22
    margin_right = 10
    
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top
    
    # Título do Gráfico
    d.add(String(width/2, height - 15, title, textAnchor='middle', fontName='Helvetica-Bold', fontSize=8, fillColor=PRIMARY))
    
    if not x or not y:
        return d
        
    x_min, x_max = 0.0, 20.0
    
    if is_residual:
        residuals = y
        max_res = max(abs(min(residuals)), abs(max(residuals)))
        y_min = -max_res - 0.5
        y_max = max_res + 0.5
        if y_max - y_min < 4:
            y_min, y_max = -2.0, 2.0
    else:
        y_min, y_max = 0.0, 20.0
        
    # Eixos e Grid
    d.add(Line(margin_left, margin_bottom, margin_left + plot_w, margin_bottom, strokeColor=DARK_TEXT, strokeWidth=0.8))
    d.add(Line(margin_left, margin_bottom, margin_left, margin_bottom + plot_h, strokeColor=DARK_TEXT, strokeWidth=0.8))
    
    # Grade de Referência
    ticks_x = [0, 5, 10, 15, 20]
    for tx in ticks_x:
        cx = margin_left + (tx / 20.0) * plot_w
        d.add(Line(cx, margin_bottom, cx, margin_bottom + plot_h, strokeColor=HexColor("#f1f2f6"), strokeWidth=0.5))
        d.add(String(cx, margin_bottom - 10, str(tx), textAnchor='middle', fontName='Helvetica', fontSize=6, fillColor=DARK_TEXT))
        
    if is_residual:
        ticks_y = [round(y_min, 1), round(y_min/2, 1), 0.0, round(y_max/2, 1), round(y_max, 1)]
    else:
        ticks_y = [0, 5, 10, 15, 20]
        
    for ty in ticks_y:
        cy = margin_bottom + ((ty - y_min) / (y_max - y_min)) * plot_h
        d.add(Line(margin_left, cy, margin_left + plot_w, cy, strokeColor=HexColor("#f1f2f6"), strokeWidth=0.5))
        d.add(String(margin_left - 4, cy - 2, f"{ty:.1f}" if is_residual else str(ty), textAnchor='end', fontName='Helvetica', fontSize=6, fillColor=DARK_TEXT))
        
    if is_residual:
        cy_zero = margin_bottom + ((0.0 - y_min) / (y_max - y_min)) * plot_h
        d.add(Line(margin_left, cy_zero, margin_left + plot_w, cy_zero, strokeColor=ERROR_COLOR, strokeWidth=1, strokeDashArray=[2, 2]))
    else:
        cx_start = margin_left + (0.0 / 20.0) * plot_w
        cy_start = margin_bottom + ((0.0 - y_min) / (y_max - y_min)) * plot_h
        cx_end = margin_left + (20.0 / 20.0) * plot_w
        cy_end = margin_bottom + ((20.0 - y_min) / (y_max - y_min)) * plot_h
        d.add(Line(cx_start, cy_start, cx_end, cy_end, strokeColor=ERROR_COLOR, strokeWidth=1, strokeDashArray=[2, 2]))
        
    n_points = len(x)
    for i in range(n_points):
        px = x[i]
        py = y[i]
        
        cx = margin_left + ((px - x_min) / (x_max - x_min)) * plot_w
        cy = margin_bottom + ((py - y_min) / (y_max - y_min)) * plot_h
        
        if margin_left <= cx <= margin_left + plot_w and margin_bottom <= cy <= margin_bottom + plot_h:
            dot_color = PRIMARY if not is_residual else SECONDARY
            d.add(Circle(cx, cy, 1.8, fillColor=dot_color, strokeColor=None))
            
    d.add(String(margin_left + plot_w/2, margin_bottom - 20, x_label, textAnchor='middle', fontName='Helvetica-Bold', fontSize=6, fillColor=DARK_TEXT))
    
    g_y = Group()
    g_y.translate(margin_left - 22, margin_bottom + plot_h/2)
    g_y.rotate(90)
    g_y.add(String(0, 0, y_label, textAnchor='middle', fontName='Helvetica-Bold', fontSize=6, fillColor=DARK_TEXT))
    d.add(g_y)
    
    return d

def draw_importance_flowable(importances, title):
    """Desenha um gráfico de barras horizontal para importância de atributos."""
    d = Drawing(480, 160)
    
    # Fundo
    d.add(Rect(0, 0, 480, 160, fillColor=colors.white, strokeColor=BORDER_COLOR, strokeWidth=0.5))
    
    margin_left = 90
    margin_right = 45
    margin_top = 22
    margin_bottom = 15
    
    plot_w = 480 - margin_left - margin_right
    plot_h = 160 - margin_top - margin_bottom
    
    # Título
    d.add(String(240, 145, title, textAnchor='middle', fontName='Helvetica-Bold', fontSize=9, fillColor=PRIMARY))
    
    top_n = min(7, len(importances))
    cell_h = plot_h / top_n
    bar_h = cell_h * 0.6
    
    max_imp = max(x["importance"] for x in importances) if importances else 1.0
    if max_imp == 0:
        max_imp = 1.0
        
    for i in range(top_n):
        item = importances[i]
        feat = item["feature"]
        imp = item["importance"]
        
        cy = margin_bottom + (top_n - 1 - i) * cell_h + (cell_h - bar_h) / 2
        bw = (imp / max_imp) * plot_w
        
        # Barra
        d.add(Rect(margin_left, cy, bw, bar_h, fillColor=SECONDARY, strokeColor=None))
        
        # Nome da variável (esquerda)
        d.add(String(margin_left - 6, cy + bar_h/2 - 3, feat, textAnchor='end', fontName='Helvetica', fontSize=8, fillColor=DARK_TEXT))
        
        # Porcentagem (direita)
        d.add(String(margin_left + bw + 4, cy + bar_h/2 - 3, f"{imp*100:.1f}%", textAnchor='start', fontName='Helvetica-Bold', fontSize=7, fillColor=DARK_TEXT))
        
    return d

# ==================== GERADOR DE PDF ====================

def create_pdf(results_file="results_scratch.json", pdf_output="relatorio_predicao_notas.pdf"):
    """Gera o relatório PDF contendo apenas gráficos e suas explicações."""
    
    if not os.path.exists(results_file):
        print(f"Erro: O arquivo de resultados '{results_file}' não foi encontrado. Execute run_analysis_scratch.py primeiro.")
        return
        
    with open(results_file, "r", encoding="utf-8") as f:
        res = json.load(f)
        
    doc = SimpleDocTemplate(
        pdf_output,
        pagesize=A4,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    styles['Normal'].textColor = DARK_TEXT
    styles['Normal'].fontSize = 9.5
    styles['Normal'].leading = 14
    
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=PRIMARY,
        spaceAfter=15,
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=DARK_TEXT,
        spaceAfter=100,
        alignment=1
    )
    
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        textColor=DARK_TEXT,
        alignment=1
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    
    story = []
    
    # ==================== CAPA ====================
    story.append(Spacer(1, 140))
    story.append(Paragraph("PREDIÇÃO DE NOTAS DE ESTUDANTES", title_style))
    story.append(Paragraph("Análise de Gráficos, Visualizações e Interpretação de Modelos", subtitle_style))
    
    story.append(Spacer(1, 100))
    
    meta_text = """
    <b>Entidade:</b> Unifucamp - Centro Universitário Mário Palmério<br/>
    <b>Curso:</b> Ciência da Computação (Ano Letivo 2026, 1º Semestre, 7º Período)<br/>
    <b>Disciplina:</b> Inteligência Artificial II<br/>
    <b>Professor:</b> Gustavo H. R. Magalhães<br/>
    <b>Alunos:</b> Lucas Firmino Rodrigues e Carlos Eduardo Mendes<br/>
    <b>Data:</b> Julho de 2026<br/>
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(PageBreak())
    
    # ==================== SEÇÃO 1: HISTOGRAMAS ====================
    story.append(Paragraph("1. Distribuição da Nota Final (G3)", h1_style))
    story.append(Paragraph("Os gráficos abaixo mostram a contagem de notas finais (G3) em bins de notas de 0 a 20. O histograma permite identificar visualmente o comportamento geral de desempenho acadêmico dos estudantes nas duas disciplinas analisadas (Matemática e Português).", styles['Normal']))
    story.append(Spacer(1, 8))
    
    hist_mat = draw_histogram_flowable(res["Matematica"]["g3_distribution"]["labels"], res["Matematica"]["g3_distribution"]["counts"], "Notas Finais (G3) - Matemática", fill_color=PRIMARY)
    hist_por = draw_histogram_flowable(res["Portugues"]["g3_distribution"]["labels"], res["Portugues"]["g3_distribution"]["counts"], "Notas Finais (G3) - Português", fill_color=SECONDARY)
    
    story.append(KeepTogether([
        Table([[hist_mat], [Spacer(1, 8)], [hist_por]], colWidths=[480], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')])
    ]))
    story.append(Spacer(1, 10))
    
    hist_exp = """
    <b>Interpretação Visual:</b><br/>
    • <b>Efeito de Evasão (Notas Zeros):</b> Há um pico notável de notas exatamente iguais a zero, sendo consideravelmente maior em Matemática do que em Português. Visualmente, isso indica uma taxa superior de reprovação direta por abandono ou faltas na disciplina de exatas. Esse comportamento atípico cria uma distribuição bimodal.<br/>
    • <b>Comportamento Central:</b> Excluindo-se os zeros, a nota de Português assemelha-se mais a uma curva normal, concentrada simetricamente em torno de 11 a 12 pontos. Já a de Matemática exibe maior achatamento (variância), indicando um rendimento estudantil mais disperso.
    """
    story.append(Paragraph(hist_exp, styles['Normal']))
    story.append(PageBreak())
    
    # ==================== SEÇÃO 2: CORRELAÇÕES ====================
    story.append(Paragraph("2. Matrizes de Correlação Linear (Heatmaps)", h1_style))
    story.append(Paragraph("Os heatmaps de correlação de Pearson ajudam a identificar a força e a direção da associação linear entre as variáveis numéricas do dataset e a nota final (G3). As cores quentes (vermelho) indicam correlação negativa e as cores frias (azul) indicam correlação positiva.", styles['Normal']))
    story.append(Spacer(1, 8))
    
    corr_mat = draw_heatmap_flowable(res["Matematica"]["correlations"]["matrix"], res["Matematica"]["correlations"]["columns"], "Matriz de Correlação - Matemática")
    corr_por = draw_heatmap_flowable(res["Portugues"]["correlations"]["matrix"], res["Portugues"]["correlations"]["columns"], "Matriz de Correlação - Português")
    
    story.append(KeepTogether([
        Table([[corr_mat], [Spacer(1, 10)], [corr_por]], colWidths=[480], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')])
    ]))
    story.append(Spacer(1, 10))
    
    corr_exp = """
    <b>Interpretação Visual:</b><br/>
    • <b>Dominância de G1 e G2:</b> Existe uma fortíssima correlação positiva (azul escuro, &gt; 0.80) entre as notas parciais (G1 e G2) e a nota final G3. Isso demonstra que o rendimento intermediário é o fator mais determinante para o resultado final.<br/>
    • <b>Fatores Demográficos e Comportamentais:</b> A variável <i>failures</i> (reprovações anteriores) mostra correlação negativa (tons de vermelho) com as notas. O tempo de estudo (<i>studytime</i>) mostra uma suave correlação positiva, enquanto o absenteísmo (<i>absences</i>) e o ato de sair com amigos (<i>goout</i>) apresentam tendências de correlação linear negativa com a nota G3.
    """
    story.append(Paragraph(corr_exp, styles['Normal']))
    story.append(PageBreak())
    
    # ==================== SEÇÃO 3: SCATTER PLOTS ====================
    story.append(Paragraph("3. Avaliação dos Modelos (Real vs. Previsto e Resíduos)", h1_style))
    story.append(Paragraph("Esta seção apresenta a qualidade de ajuste e os padrões de erro dos modelos preditivos. O gráfico <b>Real vs. Previsto</b> compara o alinhamento dos pontos em relação à reta tracejada vermelha ideal (Y=X). O <b>Gráfico de Resíduos</b> exibe a diferença entre a nota real e a predita (eixo Y) contra as predições (eixo X).", styles['Normal']))
    story.append(Spacer(1, 8))
    
    for subject in ["Matematica", "Portugues"]:
        story.append(Paragraph(f"<b>Visualização de Erros - {subject}</b>", ParagraphStyle('Sub', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, textColor=SECONDARY, spaceBefore=4, spaceAfter=4)))
        
        y_true = res[subject]["scatter_plots"]["y_test_subset"]
        y_lr = res[subject]["scatter_plots"]["predictions_lr"]
        y_rf = res[subject]["scatter_plots"]["predictions_rf"]
        
        sc_lr_real = draw_scatter_flowable(y_true, y_lr, "Nota Real", "Nota Prevista", "Regr. Linear: Real vs Previsto", is_residual=False)
        sc_rf_real = draw_scatter_flowable(y_true, y_rf, "Nota Real", "Nota Prevista", "Random Forest: Real vs Previsto", is_residual=False)
        
        res_lr = [y_true[i] - y_lr[i] for i in range(len(y_true))]
        res_rf = [y_true[i] - y_rf[i] for i in range(len(y_true))]
        
        sc_lr_res = draw_scatter_flowable(y_lr, res_lr, "Nota Prevista", "Resíduo", "Regr. Linear: Resíduos", is_residual=True)
        sc_rf_res = draw_scatter_flowable(y_rf, res_rf, "Nota Prevista", "Resíduo", "Random Forest: Resíduos", is_residual=True)
        
        sc_table = Table([
            [sc_lr_real, sc_rf_real],
            [sc_lr_res, sc_rf_res]
        ], colWidths=[240, 240])
        sc_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(KeepTogether([sc_table]))
        story.append(Spacer(1, 5))
        
    story.append(PageBreak())
    
    scatter_exp = """
    <b>Interpretação Visual:</b><br/>
    • <b>Ajuste de Real vs. Previsto:</b> A maioria dos pontos está concentrada muito próxima à reta diagonal tracejada (ideal), refletindo o alto R² dos modelos (chegando a 0.859 em Português). No entanto, nota-se uma falha de ajuste nas notas reais iguais a zero: os modelos preveem notas entre 5 e 10 para alunos que de fato obtiveram zero, indicando dificuldade de lidar com desistências repentinas.<br/>
    • <b>Gráficos de Resíduos:</b> Em sua maioria, os resíduos estão distribuídos uniformemente em torno da linha horizontal tracejada Y=0 (homocedasticidade), confirmando boa modelagem. A cauda diagonal de pontos nos resíduos representa justamente os alunos que zeraram a nota e foram previstos com notas intermediárias, criando um padrão linear de resíduo negativo no canto esquerdo.
    """
    story.append(Paragraph(scatter_exp, styles['Normal']))
    story.append(Spacer(1, 10))
    
    # ==================== SEÇÃO 4: IMPORTÂNCIA ====================
    story.append(Paragraph("4. Importância de Variáveis no Modelo", h1_style))
    story.append(Paragraph("Os gráficos abaixo ilustram os pesos relativos atribuídos às 7 variáveis mais importantes pelo estimador Random Forest para estimar as notas de Matemática e Português. A escala de importância é calculada baseada na frequência de uso dos atributos nas decisões das árvores da floresta.", styles['Normal']))
    story.append(Spacer(1, 8))
    
    imp_mat = draw_importance_flowable(res["Matematica"]["feature_importances"], "Importância dos Atributos - Matemática")
    imp_por = draw_importance_flowable(res["Portugues"]["feature_importances"], "Importância dos Atributos - Português")
    
    story.append(KeepTogether([
        Table([[imp_mat], [Spacer(1, 10)], [imp_por]], colWidths=[480], style=[('ALIGN', (0,0), (-1,-1), 'CENTER')])
    ]))
    story.append(Spacer(1, 10))
    
    imp_exp = """
    <b>Interpretação Visual:</b><br/>
    • <b>Preditor Absoluto:</b> Em Matemática, a nota parcial <b>G2</b> responde por mais de 50% de toda a importância de decisão do modelo, seguida por <b>G1</b>. Em Português, nota-se uma dependência ainda maior de <b>G2</b> (superior a 60%).<br/>
    • <b>Variáveis Não Acadêmicas:</b> A variável <i>absences</i> (faltas) e <i>failures</i> (reprovações anteriores) aparecem como preditores secundários, confirmando que o absenteísmo escolar e o histórico de dificuldades prévias são os atributos comportamentais de maior impacto negativo para prever o desempenho acadêmico final dos alunos.
    """
    story.append(Paragraph(imp_exp, styles['Normal']))
    
    # Construir documento
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Relatório PDF gerado com sucesso em '{pdf_output}'!")

if __name__ == "__main__":
    create_pdf()
